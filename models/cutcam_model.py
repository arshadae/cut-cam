import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class CUTCAMModel(BaseModel):
    """ This is a modification of the class implements CUT and FastCUT model, 
    described in the paper Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu in ECCV, 2020

    They code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    
    Modifications are made my "Arshad MA" and the task relavent modification log is provided below.

    Modifications log:
    Task 1: Distributed data parallel (DDP) training
        - Removed the use of opt.gpu_ids, as it is not used in the context of DDP.
        - Removed all instance of '.to(self.device)' as the model is already moved to the correct device in the DDP setup.
        - Changed the way the input batch is set in the model: Now it doesn't divide the batch size by the number of GPUs, as DDP handles this automatically.
        - Added  any(p.requires_grad for p in self.netF.parameters()) condition to check if the feature network requires gradients before initializing the optimizer in data_dependent_initialization declaration.

    Task 2: Single-stage, unpaired, class-attentive translator
      - Auxiliary classifier head on G's encoder for class logits (with unpaired CUT/PatchNCE + GAN losses):
        - We lazily build netC on the same feature map that the CUT already exposes via encode_only (keeps the CUT API untouched) .
      - New losses inside compute_G_loss():
        - loss_CLS: CE on logits of input and translation (class consistency).
        - loss_CAM: L1 between GradCAM maps of input vs. translation (semantic invariance).
        - Optional loss_IDT and loss_EDGE for stability on MWIR small targets.
        - Optional CAM‑weighted adversarial (weight per‑patch GAN loss by CAM importance).
    - Everything else from CUT stays (GAN + PatchNCE + identity NCE) to maintain reproducibility.
    
    
    Suggest flags for training:
    --CUT_mode CUTCAM \
    --num_classes 7 \
    --lambda_GAN 1 --lambda_NCE 1 \
    --lambda_CLS 1 --lambda_CAM 25 \
    --lambda_IDT 10 --nce_idt \
    --lambda_EDGE 5 \
    --cam_weight_adv \
    --cam_layer -1   # use last layer from --nce_layers
    
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUTCAM model  """
        parser.add_argument('--CUT_mode', type=str, default="CUTCAM", choices='(CUT, cut, FastCUT, fastcut, CUTCAM, cutcam)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="flip-equivariance regularization (FastCUT)")

        # ---- NEW: class-attentive/XAI options ----
        parser.add_argument('--num_classes', type=int, default=7, help='number of classes for auxiliary head')
        parser.add_argument('--lambda_CAM', type=float, default=25.0, help='weight for CAM invariance loss')
        parser.add_argument('--lambda_CLS', type=float, default=1.0, help='weight for classification consistency')
        parser.add_argument('--lambda_IDT', type=float, default=0.0, help='weight for identity loss on real B')
        parser.add_argument('--lambda_EDGE', type=float, default=0.0, help='weight for edge-preservation loss')
        parser.add_argument('--cam_layer', type=int, default=-1,
                            help='index into nce_layers list to take CAM from; -1 means last entry')
        parser.add_argument('--cam_weight_adv', type=util.str2bool, nargs='?', const=True, default=False,
                            help='weight adversarial loss per patch by CAM importance')
        parser.add_argument('--cam_eps', type=float, default=1e-6, help='numerical epsilon for CAM normalization')


        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower()  in ("cut", "cutcam"):
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # The training/test scripts will call <BaseModel.get_current_losses>

        # track/capture extra losses
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'CLS', 'CAM', 'IDT', 'EDGE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.cam_layer_idx = self.nce_layers[self.opt.cam_layer] if self.opt.cam_layer != -1 else self.nce_layers[-1]
        # self.opt.cam_layer → user-chosen index into the NCE layer list.
        # cam_layer_idx → the actual resolved feature layer ID (e.g., 8, 16) that gets fed to the generator for GradCAM maps.

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        # models to save/load
        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'C'] #NEW: add C (auxiliary classifier head)
        else:  # during inference, only G is needed
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt)

        # Auxiliary classifier head will be created lazily when we first see encoder feature shape
        self.netC = None
        self._built_C = False


        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            self.criterionNCE = [PatchNCELoss(opt) for _ in self.nce_layers]
            self.criterionCE = nn.CrossEntropyLoss()
            self.criterionL1 = nn.L1Loss()
            
            # optimizers (G and D now; C and F will be added lazily)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    # ------ util: lazy build of classifier head on encoder features ------
    def _build_netC_if_needed(self, feat_tensor):
        if self._built_C:
            return
        in_ch = feat_tensor.shape[1]
        # simple head: 1x1 conv -> GAP -> linear
        self.netC = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, self.opt.num_classes, bias=True),
        )

        # Make sure the classifier is on the same device as the features
        self.netC = self.netC.to(feat_tensor.device)

        # <<< IMPORTANT: put netC on the same device as the features >>>
        self.netC.to(feat_tensor.device)

        if self.isTrain:
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_C)
        self._built_C = True


    # ------ data init (kept; plus class labels if provided) ------
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
       
        # build netC based on encoder feature shape
        with torch.no_grad():
            fA = self.netG(self.real_A, [self.cam_layer_idx], encode_only=True)[0]
        self._build_netC_if_needed(fA)

        if self.opt.isTrain:
            # prime optimizers (kept behavior)
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0 and any(p.requires_grad for p in self.netF.parameters()):
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    # ------ standard CUT loop (kept) ------
    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G (and C, and F)
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if hasattr(self, 'opimizer_F') and self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        if hasattr(self, 'optimizer_C'):
            self.optimizer_C.zero_grad()
    
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()

        self.optimizer_G.step()
        if hasattr(self, 'optimizer_F') and self.opt.netF == 'mlp_sample':
            # update F only if it is a sample-based MLP
            # (e.g. PatchSampleF)
            self.optimizer_F.step()
        if hasattr(self, 'optimizer_C'):
            # update C only if it is defined
            self.optimizer_C.step()

    # ------ inputs ------
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B']
        self.real_B = input['B' if AtoB else 'A']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # Optional class labels for A-domain (int64). If absent, we will skip CE and fallback to pseudo-labels.
        self.cls = input.get('cls', None)


    # ------ forward pass ------
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    # ------ D loss (kept; optionally CAM-weighted via compute_G_loss) ------
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    # ------ G total loss ------
    def compute_G_loss_old(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G



    # ------ G total loss ------
    def compute_G_loss(self):
        fake = self.fake_B

        # 1) adversarial (optionally CAM-weighted)
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)  # [B,1,Hd,Wd]
            adv_raw = self.criterionGAN(pred_fake, True)  # per-patch
            if self.opt.cam_weight_adv and self._built_C:
                # upsample CAM(fake) to D's patch resolution and weight the adv loss
                cam_fake = self._compute_cam_map(fake, use_input=False)  # [B,1,H,W]
                cam_w = F.interpolate(cam_fake, size=adv_raw.shape[-2:], mode='bilinear', align_corners=False)
                cam_w = cam_w / (cam_w.mean(dim=[2,3], keepdim=True) + self.opt.cam_eps)  # normalize average weight ~1
                self.loss_G_GAN = (adv_raw * cam_w).mean() * self.opt.lambda_GAN
            else:
                self.loss_G_GAN = adv_raw.mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # 2) CUT PatchNCE (kept)
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # 3) Classification consistency (optional; needs labels or pseudo-labels)
        self.loss_CLS = 0.0
        if self._built_C:
            # encoder features for A and fake_B at the chosen CAM layer
            fA = self.netG(self.real_A, [self.cam_layer_idx], encode_only=True)[0]
            fF = self.netG(self.fake_B, [self.cam_layer_idx], encode_only=True)[0]
            # build head lazily if not yet done
            self._build_netC_if_needed(fA)
            logits_A = self._class_logits(fA)
            logits_F = self._class_logits(fF)
            # labels: prefer provided labels; else pseudo-label from logits_A
            if self.cls is not None:
                y = self.cls.long()
            else:
                y = torch.argmax(logits_A.detach(), dim=1)
            self.loss_CLS = (self.criterionCE(logits_A, y) + self.criterionCE(logits_F, y)) * 0.5 * self.opt.lambda_CLS

        # 4) GradCAM invariance (align CAM maps between input and output)
        self.loss_CAM = 0.0
        if self._built_C and self.opt.lambda_CAM > 0.0:
            cam_A = self._compute_cam_map(self.real_A, use_input=True)   # [B,1,H,W]
            cam_F = self._compute_cam_map(self.fake_B, use_input=False)  # [B,1,H,W]
            self.loss_CAM = F.l1_loss(cam_A, cam_F) * self.opt.lambda_CAM

        # 5) Identity loss (optional)
        self.loss_IDT = 0.0
        if self.opt.lambda_IDT > 0.0 and self.opt.nce_idt:
            self.loss_IDT = self.criterionL1(self.idt_B, self.real_B) * self.opt.lambda_IDT

        # 6) Edge preservation (optional)
        self.loss_EDGE = 0.0
        if self.opt.lambda_EDGE > 0.0:
            self.loss_EDGE = self._edge_loss(self.fake_B, self.real_A) * self.opt.lambda_EDGE

        # total
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_CLS + self.loss_CAM + self.loss_IDT + self.loss_EDGE
        return self.loss_G



    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and getattr(self, 'flipped_for_equivariance', False):
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    # ------ helpers: classifier logits + GradCAM map ------
    def _class_logits(self, f):  # f: [B,C,H,W]
        return self.netC(f)

    @torch.no_grad()
    def _minmax_norm(self, m):  # m: [B,1,H,W]
        B = m.shape[0]
        m = m.view(B, -1)
        m_min = m.min(dim=1, keepdim=True)[0]
        m_max = m.max(dim=1, keepdim=True)[0]
        m = (m - m_min) / (m_max - m_min + self.opt.cam_eps)
        return m.view(-1, 1, *self.fake_B.shape[-2:])

    def _compute_cam_map(self, x, use_input=True):
        """
        Differentiable GradCAM on G's encoder (chosen cam_layer_idx) w.r.t. class y.
        If labels present, we use them; otherwise we use argmax of logits(x).
        Returns a [B,1,H,W] map normalized to [0,1] with the spatial size of x.
        """
        # Get encoder feature without tracking graph (we'll treat it as a leaf)
        with torch.no_grad():
            f = self.netG(x, [self.cam_layer_idx], encode_only=True)[0]  # [B,C,Hf,Wf]
        self._build_netC_if_needed(f)
        # Everything below must run with grad even in eval/no_grad contexts
        with torch.enable_grad():
            f = f.detach().requires_grad_(True)  # make f a leaf that requires grad
            logits = self._class_logits(f)  # [B,K]
            if self.cls is not None and use_input:
                y = self.cls.long()
            else:
                y = torch.argmax(logits.detach(), dim=1)

            # select target logit per sample
            selected = logits.gather(1, y.view(-1, 1)).sum()  # scalar
            grads = torch.autograd.grad(
                outputs=selected,
                inputs=f,
                create_graph=False,         # no need to build higher-order graph at test time
                retain_graph=True
            )[0]  # [B,C,Hf,Wf]


        # Grad-CAM: weights = GAP(grads), cam = ReLU(sum_c w_c * f_c)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = F.relu((weights * f).sum(dim=1, keepdim=True))  # [B,1,Hf,Wf]

        # upsample & normalize
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # per-sample min-max normalization to [0,1]
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]
        cam = ((cam_flat - cam_min) / (cam_max - cam_min + self.opt.cam_eps)).view(B, 1, x.shape[-2], x.shape[-1])
        return cam
    
    # A tiny public wrapper so users don’t call a “private” method
    def get_cam_maps(self, x, y_hat):
        return self._compute_cam_map(x, use_input=True), self._compute_cam_map(y_hat, use_input=False)



    # ------ optional edge loss (Sobel) ------
    def _edge_loss(self, y_hat, x):
        def sobel(img):
            # expects [B,C,H,W]
            kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
            ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
            if img.shape[1] > 1:
                # convert to luma for edge comparison
                img = img.mean(dim=1, keepdim=True)
            gx = F.conv2d(img, kx, padding=1)
            gy = F.conv2d(img, ky, padding=1)
            g = torch.sqrt(gx*gx + gy*gy + 1e-6)
            return g
        return F.l1_loss(sobel(y_hat), sobel(x))