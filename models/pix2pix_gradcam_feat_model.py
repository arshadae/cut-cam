import torch
from .base_model import BaseModel
from . import networks
from torchvision.models import resnet18
import torch.nn.functional as F
from captum.attr import LayerGradCam
from checkpoints.classifiers.resnet18_mwir import get_trained_resnet18
import matplotlib.pyplot as plt


class Pix2PixGradCamFeatModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--resnet18_path', type=str, default='', help='path to pretrained ResNet18 for Grad-CAM')
            parser.add_argument('--lambda_gradcam', type=float, default=100.0, help='weight for Grad-CAM loss')
            parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for classifier feature matching loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if opt.lambda_gradcam > 0 and opt.resnet18_path != '':
            print(f"Loading CAM classifier from {opt.resnet18_path}")
            self.classifier = get_trained_resnet18(weights_path="checkpoints/classifiers/resnet18_mwir.pth", num_classes=7).to(self.device)
            self.classifier.requires_grad_(False)
            self.classifier.eval()
            for param in self.classifier.parameters():
                param.requires_grad = False

            self.cam_target_layer = self.classifier.layer4[1]
            self.cam_model = LayerGradCam(self.classifier, self.cam_target_layer)
            self.lambda_gradcam = opt.lambda_gradcam
        else:
            print("CAM is disabled or resnet18 weights not provided.")
            self.classifier = None
            self.cam_target_layer = None
            self.cam_model = None
            self.lambda_gradcam = 0.0

        self.lambda_feat = opt.lambda_feat

        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if self.lambda_gradcam > 0:
            self.loss_names.append('G_gradcam')
        self.loss_names.append('G_feat')

        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.lambda_gradcam > 0:
            self.visual_names += ['gradcam_fake_B', 'gradcam_real_B']

        self.model_names = ['G', 'D'] if self.isTrain else ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGradCAM = torch.nn.MSELoss()
            self.criterionFeature = torch.nn.MSELoss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        if self.cam_model is not None and self.lambda_gradcam > 0:
            self.gradcam_real_B, _ = self.compute_batch_gradcam_heatmaps(self.real_B)
            self.gradcam_fake_B, _ = self.compute_batch_gradcam_heatmaps(self.fake_B)
            self.gradcam_real_B.detach()
            self.gradcam_fake_B.detach()

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def compute_batch_gradcam_heatmaps(self, images, target_classes=None):
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
            min_val, max_val = heatmap.min(), heatmap.max()
            if max_val != min_val:
                heatmap = (heatmap - min_val) / (max_val - min_val)
            heatmaps.append(heatmap)

        heatmaps = torch.cat(heatmaps, dim=0)
        return heatmaps, target_classes

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        if self.cam_model is not None and self.lambda_gradcam > 0:
            gradcam_real_B, target_classes = self.compute_batch_gradcam_heatmaps(self.real_B)
            gradcam_fake_B, _ = self.compute_batch_gradcam_heatmaps(self.fake_B, target_classes)
            self.loss_G_gradcam = self.criterionGradCAM(gradcam_fake_B, gradcam_real_B) * self.lambda_gradcam
        else:
            self.loss_G_gradcam = 0.0

        with torch.no_grad():
            feat_real = self.classifier.avgpool(
                self.classifier.layer4(
                    self.classifier.layer3(
                        self.classifier.layer2(
                            self.classifier.layer1(self.real_B)
                        )
                    )
                )
            )
        feat_fake = self.classifier.avgpool(
            self.classifier.layer4(
                self.classifier.layer3(
                    self.classifier.layer2(
                        self.classifier.layer1(self.fake_B)
                    )
                )
            )
        )
        feat_real = torch.flatten(feat_real, 1)
        feat_fake = torch.flatten(feat_fake, 1)
        self.loss_G_feat = self.criterionFeature(feat_fake, feat_real) * self.lambda_feat

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_gradcam + self.loss_G_feat
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
