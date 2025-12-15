import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch import autograd as autograd
import math


"""
Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2*): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7*): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16*): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace)
      (34*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""


# --------------------------------------------
# Perceptual loss
# --------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2,7,16,25,34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # 自动将单通道灰度图扩展为3通道，以兼容 VGG19
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """

    def __init__(self, feature_layer=[2,7,16,25,34], weights=[0.1,0.1,1.0,1.0,1.0], lossfn_type='l1', use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

# --------------------------------------------
# GAN loss: gan, ragan
# --------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == 'softplusgan':
            def softplusgan_loss(input, target):
                # target is boolean
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'softplusgan']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

# --------------------------------------------
# DirectionalGradientLoss
# --------------------------------------------
class DirectionalGradientLoss(nn.Module):
    def __init__(self, weight_x=1.0, weight_y=1.0):
        """
        Args:
            weight_x: 水平梯度 (检测纵向边缘) 的权重
            weight_y: 垂直梯度 (检测横向边缘) 的权重
        说明:
            对于【横向条纹】，建议加大 weight_y，强迫网络恢复正确的垂直梯度。
        """
        super(DirectionalGradientLoss, self).__init__()
        self.weight_x = weight_x
        self.weight_y = weight_y
        
        # 定义 Sobel 算子
        # Kernel X: 计算 dI/dx (检测垂直线条)
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        # Kernel Y: 计算 dI/dy (检测水平线条)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        
        # 注册为 buffer：不需要训练参数，且会自动随模型迁移到 GPU
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, target):
        # 自动适配输入通道数 (Batch, C, H, W)
        b, c, h, w = pred.shape
        
        # 扩展 Kernel 以适配多通道输入 (Depthwise Convolution)
        # 这样无论是单通道 CT 还是 3 通道都能跑
        kernel_x: torch.Tensor = self.kernel_x  # type: ignore
        kernel_y: torch.Tensor = self.kernel_y  # type: ignore
        kx = kernel_x.expand(c, 1, 3, 3)
        ky = kernel_y.expand(c, 1, 3, 3)

        # 1. 计算预测图的梯度 (groups=c 保证各通道独立计算)
        pred_gx = F.conv2d(pred, kx, padding=1, groups=c)
        pred_gy = F.conv2d(pred, ky, padding=1, groups=c)
        
        # 2. 计算真值图的梯度
        target_gx = F.conv2d(target, kx, padding=1, groups=c)
        target_gy = F.conv2d(target, ky, padding=1, groups=c)
        
        # 3. 计算 L1 Loss (L1 比 MSE 对边缘更敏感且不模糊)
        loss_x = F.l1_loss(torch.abs(pred_gx), torch.abs(target_gx))
        loss_y = F.l1_loss(torch.abs(pred_gy), torch.abs(target_gy))
        
        return self.weight_x * loss_x + self.weight_y * loss_y

# --------------------------------------------
# DirectionalTV loss
# --------------------------------------------
class DirectionalTVLoss(nn.Module):
    def __init__(self, weight_h=1.0, weight_w=1.0):
        """
        Args:
            weight_h: 惩罚垂直方向的差分 (Height wise) -> 用于消除【横向条纹】
            weight_w: 惩罚水平方向的差分 (Width wise)  -> 用于消除【纵向条纹】
        """
        super(DirectionalTVLoss, self).__init__()
        self.weight_h = weight_h
        self.weight_w = weight_w

    def forward(self, x):
        # 使用 L1 (abs) 代替 L2 (pow)，L1 去伪影更干净，不容易产生模糊
        
        # 计算垂直方向梯度 (对应横向条纹的边缘)
        # 伪影在上下方向跳变剧烈，h_tv 会很大
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        
        # 计算水平方向梯度
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        
        return self.weight_h * h_tv + self.weight_w * w_tv


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss



def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.
        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty
