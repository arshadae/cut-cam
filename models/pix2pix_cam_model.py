import torch
from .base_model import BaseModel
from . import networks
from torchvision.models import resnet18
import torch.nn.functional as F



class Pix2PixCAMModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            # ===== Grad-CAM additions =====
            parser.add_argument('--resnet18_path', type=str, default='', help='path to pretrained ResNet18 for Grad-CAM')
            parser.add_argument('--lambda_gradcam', type=float, default=10.0, help='weight for Grad-CAM loss')

        return parser

    def __init__(self, opt):
        """Initialize the Pix2_Pix_CAM_Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # Always define lambda_gradcam early
        # self.lambda_gradcam = opt.lambda_gradcam if (opt.lambda_gradcam > 0 and opt.resnet18_path != '') else 0.0
        # self.cam_model = None
        # self.cam_target_layer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ===== CAM additions =====
        if opt.lambda_gradcam > 0 and opt.resnet18_path != '':
            print(f"Loading CAM classifier from {opt.resnet18_path}")
            self.cam_model = resnet18(pretrained=False) # we don’t want to load the default ImageNet weights
            # Manually adjust fc layer to match saved model
            self.cam_model.fc = torch.nn.Linear(self.cam_model.fc.in_features, 7)  # 7 = number of classes in our pretrained model

            # Load the weights
            state_dict = torch.load(opt.resnet18_path, map_location=self.device)
            self.cam_model.load_state_dict(state_dict)
            self.cam_model.eval().to(self.device)
            self.cam_target_layer = self.cam_model.layer4  # target layer for Grad-CAM
            self.lambda_gradcam = opt.lambda_gradcam

            self.cam_model.eval().to(self.device)
            for param in self.cam_model.parameters():
                param.requires_grad = True

        else:
            print("CAM is disabled or resnet18 weights not provided.")
            self.cam_model = None
            self.cam_model = None
            self.cam_target_layer = None
            self.lambda_gradcam = 0.0



        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # ===== Grad-CAM additions =====
        if opt.lambda_gradcam > 0 and opt.resnet18_path != '':
            self.loss_names.append('G_gradcam')

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.lambda_gradcam > 0:
            self.visual_names += ['real_cam', 'fake_cam']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']            

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG.to(self.device)


        if self.isTrain: # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD.to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

        # Compute Grad-CAM heatmaps here for visualization
        if self.cam_model is not None and self.lambda_gradcam > 0:
            self.real_cam = self.compute_gradcam_heatmap(self.real_B).detach()
            self.fake_cam = self.compute_gradcam_heatmap(self.fake_B).detach()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)   # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    # ===== Grad-CAM helper method =====
    def compute_gradcam_heatmap(self, image, target_class=None):
        assert self.cam_model is not None and self.cam_target_layer is not None, "Grad-CAM model or target layer not set."

        image = image.clone().to(self.device).requires_grad_(True)

        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        handle_fw = self.cam_target_layer.register_forward_hook(forward_hook)
        handle_bw = self.cam_target_layer.register_full_backward_hook(backward_hook)

        # Ensure gradients are enabled during forward
        with torch.enable_grad():
            output = self.cam_model(image)  # shape: (batch_size, num_classes)

            # Determine target classes for Grad-CAM
            if target_class is None:
                # Default: use predicted classes per image in batch
                class_idx = output.argmax(dim=1)
            else:
                # If target_class is int, create a tensor of that class repeated for the batch
                if isinstance(target_class, int):
                    class_idx = torch.full((image.size(0),), target_class, dtype=torch.long, device=self.device)
                else:
                    # Else assume target_class is a tensor with per-image target classes
                    class_idx = target_class.to(self.device)

            # Select scores for the target class per batch item and sum them
            scores = output[torch.arange(image.size(0)), class_idx].sum()

        self.cam_model.zero_grad()
        scores.backward(retain_graph=True)

        handle_fw.remove()
        handle_bw.remove()

        grad = gradients[0]  # (B, C, H, W)
        act = activations[0] # (B, C, H, W)

        pooled_grad = torch.mean(grad, dim=[2, 3], keepdim=True)  # global average pooling of gradients
        weighted_act = act * pooled_grad
        heatmap = weighted_act.sum(dim=1, keepdim=True)  # weighted combination along channels
        heatmap = F.relu(heatmap)

        # Upsample heatmap to input image size
        heatmap = F.interpolate(heatmap, size=image.shape[2:], mode='bilinear', align_corners=False)

        # Normalize heatmap to [0, 1]
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        if heatmap_max != heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        return heatmap



    def enhance_heatmap(self, tensor):
        tensor = tensor.squeeze().cpu().detach().numpy()
        plt.imshow(tensor, cmap='jet')
        plt.colorbar()
        plt.axis('off')
        # plt.savefig(name, bbox_inches='tight')
        tensor_en = tensor
        plt.close()
        return tensor_en

        # enhance_heatmap(heatmap_fake[0], f"{SAVE_DIR}/heatmap_fake_epoch_{epoch+1}.png")
        # enhance_heatmap(heatmap_real[0], f"{SAVE_DIR}/heatmap_real_epoch_{epoch+1}.png")

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Third CAM(A) = CAM(B)
        # ===== Grad-CAM loss addition =====
        if self.cam_model is not None and self.lambda_gradcam > 0:
            # with torch.no_grad():
            real_cam = self.compute_gradcam_heatmap(self.real_B)
            fake_cam = self.compute_gradcam_heatmap(self.fake_B)

            self.loss_G_gradcam = self.criterionL1(fake_cam, real_cam) * self.lambda_gradcam

            # Enhance heatmap for proper visualization
        
            # real_cam[0] = self.enhance_heatmap(real_cam[0])


        else:
            self.loss_G_gradcam = 0.0
            # self.real_cam = torch.zeros_like(self.real_B[:, :1, :, :])  # dummy grayscale
            # self.fake_cam = torch.zeros_like(self.fake_B[:, :1, :, :])

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_gradcam
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                              # Compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)     # Enable backprop for D
        self.optimizer_D.zero_grad()                # Set D's gradients to zero
        self.backward_D()                           # Calculate gradients for D
        self.optimizer_D.step()                     # Update D's weights

        # update G
        self.set_requires_grad(self.netD, False)    # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()                # Set G's gradients to zero
        self.backward_G()                           # Calculate graidents for G
        self.optimizer_G.step()                     # Update G's weights
