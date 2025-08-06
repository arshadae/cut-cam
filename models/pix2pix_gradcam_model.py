import torch
from .base_model import BaseModel
from . import networks
from torchvision.models import resnet18
import torch.nn.functional as F
from captum.attr import LayerGradCam
from checkpoints.classifiers.resnet18_mwir import get_trained_resnet18  # Load your trained classifier
import matplotlib.pyplot as plt



class Pix2PixGradCAMModel(BaseModel):
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

        For Pix2Pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use LSGAN for GAN loss, UNet with batchnorm, and aligned datasets.

        For Pix2Pix_GradCAM, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1 + lambda_gradcam * (||GradCAM_heatmap(G(A)) - GradCAM_heatmap(B)||_2)^2
        By default, we use LSGAN for GAN loss, UNet with batchnorm, and aligned datasets for Pix2Pix part.
        Additionally, we use ResNet18 pretrained with target classes as classifier to generate normalized GradCAM_heatmaps.

        Set --lambda_gradcam to 0 to run Pix2Pix without GradCAM.

        Loss Interpretation (LSGAN):
        LSGAN uses least-squares loss instead of BCE as in vanilla GAN. The ideal values are:
        D_real = (netD(real) − 1)² → should be small when netD(real) ≈ 1
        D_fake = (netD(fake) − 0)² → should be small when netD(fake) ≈ 0
        G_GAN = (netD(fake) − 1)² → should be small when netG fools netD (netD(fake) ≈ 1)

        """
        # changing the default values to match the pix2pix (https://phillipi.github.io/pix2pix/) with lsgan GAN loss and GradCAM settings.
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            # ===== Grad-CAM additions =====
            parser.add_argument('--resnet18_path', type=str, default='', help='path to pretrained ResNet18 for Grad-CAM')
            parser.add_argument('--lambda_gradcam', type=float, default=100.0, help='weight for Grad-CAM loss')
            
        return parser

    def __init__(self, opt):
        """Initialize the Pix2_Pix_CAM_Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ===== CAM additions =====
        if opt.lambda_gradcam > 0 and opt.resnet18_path != '':
            print(f"Loading CAM classifier from {opt.resnet18_path}")
            self.classifier = get_trained_resnet18(weights_path="checkpoints/classifiers/resnet18_mwir.pth", num_classes=7).to(self.device)
            # self.classifier = get_trained_resnet18(weights_path=self.resnet18_path, num_classes=7).to(self.device)
            self.classifier.requires_grad_(False)
            
            self.classifier.eval()
            for param in self.classifier.parameters():
                param.requires_grad = False

            self.cam_target_layer = self.classifier.layer4[1]  # Modify index if needed
            self.cam_model = LayerGradCam(self.classifier, self.cam_target_layer)
            self.lambda_gradcam = opt.lambda_gradcam

        else:
            print("CAM is disabled or resnet18 weights not provided.")
            self.classifier = None
            self.cam_target_layer = None
            self.cam_model = None
            self.lambda_gradcam = 0.0



        # specify the training losses you want to print out. 
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # ===== Grad-CAM additions =====
        if opt.lambda_gradcam > 0 and opt.resnet18_path != '':
            self.loss_names.append('G_gradcam')

        self.visual_names = ['real_A', 'fake_B']
        if not self.isTrain and self.lambda_gradcam > 0:
            self.visual_names.append('gradcam_fake_B')
            # gradcam_real_B will only be added dynamically if available during forward
            # Or you can just avoid visualizing it when real_B is None
        if self.isTrain:
            self.visual_names.append('real_B')
            if self.lambda_gradcam > 0:
                self.visual_names += ['gradcam_fake_B', 'gradcam_real_B']


        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']            

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)
        

        if self.isTrain: # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGradCAM = torch.nn.MSELoss()

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
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input.get('B' if AtoB else 'A', None)
        if self.real_B is not None:
            self.real_B = self.real_B.to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

        if self.cam_model is not None and self.lambda_gradcam > 0:
            if self.real_B is not None:
                self.gradcam_real_B, _ = self.compute_batch_gradcam_heatmaps(self.real_B)
            else:
                self.gradcam_real_B = torch.zeros_like(self.fake_B)  # dummy tensor or skip visualization
            self.gradcam_fake_B, _ = self.compute_batch_gradcam_heatmaps(self.fake_B)
            self.gradcam_fake_B.detach_()
            if self.gradcam_real_B is not None:
                self.gradcam_real_B.detach_()

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
    def compute_batch_gradcam_heatmaps(self, images, target_classes=None):
        """
        Compute Grad-CAM heatmaps for a batch of images.

        Args:
            images (Tensor): Batch of input images (B, C, H, W)
            classifier (nn.Module): The model used for classification
            gradcam (LayerGradCam): Captum LayerGradCam instance
            target_classes (Tensor or None): Optional tensor of target class indices (B,)
            device (str): Device to run computations on

        Returns:
            heatmaps (Tensor): Heatmaps of shape (B, 1, H, W)
            target_classes (Tensor): Target classes used for Grad-CAM
        """
        images = images.to(self.device)

        if target_classes is None:
            target_classes = []
            self.classifier.eval()
            with torch.no_grad():
                for i in range(images.size(0)):
                    logits = self.classifier(images[i].unsqueeze(0))
                    target_class = logits.argmax(dim=1).item()
                    target_classes.append(target_class)
            target_classes = torch.tensor(target_classes, device=self.device)

        heatmaps = []
        for i in range(images.size(0)):
            heatmap = self.cam_model.attribute(images[i].unsqueeze(0), target=target_classes[i])
            heatmap = F.interpolate(heatmap, size=images.shape[2:], mode='bilinear', align_corners=False)
            heatmap = F.relu(heatmap)

            # Normalize to [0, 1]
            min_val, max_val = heatmap.min(), heatmap.max()
            if max_val != min_val:
                heatmap = (heatmap - min_val) / (max_val - min_val)

            heatmaps.append(heatmap)

        heatmaps = torch.cat(heatmaps, dim=0)  # shape: (B, 1, H, W)
        return heatmaps, target_classes


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # print(self.real_A.device, self.real_B.device, self.fake_B.device)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Third CAM(A) = CAM(B)
        # ===== Grad-CAM loss addition =====
        if self.cam_model is not None and self.lambda_gradcam > 0:
            
            gradcam_real_B, target_classes = self.compute_batch_gradcam_heatmaps(self.real_B)
            gradcam_fake_B, _ = self.compute_batch_gradcam_heatmaps(self.fake_B, target_classes)           
            self.loss_G_gradcam = self.criterionGradCAM(gradcam_fake_B, gradcam_real_B) * self.lambda_gradcam



        else:
            self.loss_G_gradcam = 0.0


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
