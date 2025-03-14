import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import numpy as np
import cv2

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

######################### CA attention
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        # self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1,
        #                           bias=False)
        self.conv_1x1_h = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                                  bias=False)

        self.conv_1x1_w = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # batch_size, c, h, w
        _, _, h, w = x.size()

        # batch_size, c, h, w => batch_size, c, h, 1 => batch_size, c, 1, h
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # batch_size, c, h, w => batch_size, c, 1, w
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_split_h = self.conv_1x1_h(x_h)

        x_cat_conv_split_w = self.conv_1x1_w(x_w)

        # batch_size, c / r, 1, h => batch_size, c / r, h, 1 => batch_size, c, h, 1
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # batch_size, c / r, 1, w => batch_size, c, 1, w
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
#############################

class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.CA = CA_Block(128)
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.local_sobel = HGE(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.CA(x)
        local = self.local2(x) + self.local1(x) + self.local_sobel(x)



        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out



class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat
#Hybrid Gradient Enhancement Module
class HGE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(HGE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.fc = nn.Conv2d(out_channels*5, out_channels, 1, 1, 0, bias=True)
        # 创建一个可训练的卷积核参数，形状为 (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        # 初始化第一行的权重为可训练参数
        nn.init.xavier_uniform_(self.weight[:, :, :, :])

    def forward(self, x):
        # key = 'fc.weight'
        # weights_t = self.state_dict()[key]
        # w1 = weights_t[0, 0, :, :]
        # print(w1)
        # 复制第一行的权重到第三行并取相反数
        b,c,h,w=self.weight.size()
        weight = self.weight.clone()

        Sob0=weight[:, :, 0, 0]
        Sob45=weight[:, :, 0, 1]
        Sob90=weight[:, :, 0, 2]
        Sob135=weight[:, :, 1, 0]

        GL=weight[:, :, 2, 2]

        weight_0= torch.zeros(b, c, h, w).cuda()

        weight_0[:, :,0, 0] = Sob0
        weight_0[:, :,0, 1] = Sob0*2
        weight_0[:, :,0, 2] = Sob0
        weight_0[:, :,1, 0] = 0
        weight_0[:, :,1, 1] = 0
        weight_0[:, :,1, 2] = 0
        weight_0[:, :,2, 0] = -Sob0
        weight_0[:, :,2, 1] = -Sob0*2
        weight_0[:, :,2, 2] = -Sob0
        x_0=F.conv2d(x, weight_0, stride=self.stride, padding=self.padding)

        weight_45=torch.zeros(b,c,h,w).cuda()
        # weight_45=torch.zeros(b,c,h,w)
        weight_45[:, :,0, 0] = 2*Sob45
        weight_45[:, :,0, 1] = Sob45
        weight_45[:, :,0, 2] = 0
        weight_45[:, :,1, 0] = Sob45
        weight_45[:, :,1, 1] = 0
        weight_45[:, :,1, 2] = -Sob45
        weight_45[:, :,2, 0] = 0
        weight_45[:, :,2, 1] = -Sob45
        weight_45[:, :,2, 2] = -Sob45*2
        x_45=F.conv2d(x, weight_45, stride=self.stride, padding=self.padding)

        weight_90 = torch.zeros(b, c, h, w).cuda()
        weight_90[:, :, 0, 0] = Sob90
        weight_90[:, :, 0, 1] = 0
        weight_90[:, :, 0, 2] = -Sob90
        weight_90[:, :, 1, 0] = Sob90*2
        weight_90[:, :, 1, 1] = 0
        weight_90[:, :, 1, 2] = -Sob90*2
        weight_90[:, :, 2, 0] = Sob90
        weight_90[:, :, 2, 1] = 0
        weight_90[:, :, 2, 2] = -Sob90
        x_90=F.conv2d(x, weight_90, stride=self.stride, padding=self.padding)

        weight_135 = torch.zeros(b, c, h, w).cuda()
        weight_135[:, :, 0, 0] = 0
        weight_135[:, :, 0, 1] = -Sob135
        weight_135[:, :, 0, 2] = -Sob135*2
        weight_135[:, :, 1, 0] = Sob135
        weight_135[:, :, 1, 1] = 0
        weight_135[:, :, 1, 2] = -Sob135
        weight_135[:, :, 2, 0] = Sob135*2
        weight_135[:, :, 2, 1] = Sob135
        weight_135[:, :, 2, 2] = 0
        x_135=F.conv2d(x, weight_135, stride=self.stride, padding=self.padding)

        a1 =  GL * 4
        a2 = -a1 / 8
        a3 = -a1 / 16

        # kernel_Gaussian_Laplacian=torch.zeros(b,c,5,5)
        kernel_Gaussian_Laplacian=torch.zeros(b,c,5,5).cuda()
        kernel_Gaussian_Laplacian[:, :,0, 2] = a3
        kernel_Gaussian_Laplacian[:, :,1, 1] = a3
        kernel_Gaussian_Laplacian[:, :,1, 2] = a2
        kernel_Gaussian_Laplacian[:, :,1, 3] = a3
        kernel_Gaussian_Laplacian[:, :,2, 0] = a3
        kernel_Gaussian_Laplacian[:, :,2, 1] = a2
        kernel_Gaussian_Laplacian[:, :,2, 2] = a1
        kernel_Gaussian_Laplacian[:, :,2, 3] = a2
        kernel_Gaussian_Laplacian[:, :,2, 4] = a3
        kernel_Gaussian_Laplacian[:, :,3, 1] = a3
        kernel_Gaussian_Laplacian[:, :,3, 2] = a2
        kernel_Gaussian_Laplacian[:, :,3, 3] = a3
        kernel_Gaussian_Laplacian[:, :,4, 2] = a3

        x_GL=F.conv2d(x, kernel_Gaussian_Laplacian, stride=self.stride, padding=2)

        x_out= torch.cat((x_0, x_45,x_90,x_135,x_GL), dim=1)
        x_out=self.fc(x_out)


        # 使用F.conv2d进行卷积操作
        return x_out


# Gradient decoupling masks generation
class GDM(nn.Module):
    def __init__(self):
        super(GDM, self).__init__()
        # self.weight = nn.Parameter(torch.Tensor(100,100))
        # nn.init.xavier_uniform_(self.weight[:,:])
        self.weight = nn.Parameter(torch.Tensor(1,5))
        # self.weight = nn.Parameter(torch.Tensor([1,2,3,4,5]))
        # self.weight2 = nn.Parameter(torch.Tensor([1,2,3,-4.1,-5]))
        nn.init.xavier_uniform_(self.weight[:,:])
        self.conv = ConvBN(3, 5,kernel_size=7,stride=2)


    def forward(self, x):
        kernel_size = (3, 3)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        average_pooling1 = torch.nn.AvgPool2d(kernel_size, stride=2, padding=padding)
        x = average_pooling1(x)
        image_a = torch.mean(x, dim=1, keepdim=True)

        ###################weight_threshold
        weight_att = self.conv(x)
        weight_att = torch.mean(weight_att, dim=(2, 3), keepdim=True)
        weight_att = torch.squeeze(weight_att, dim=2)
        weight_att = torch.squeeze(weight_att, dim=2)
        weight_att = torch.tanh(weight_att)

        ############################
        weight = self.weight
        GL = weight[0,0]
        a1 = GL * 4
        a2 = -a1 / 8
        a3 = -a1 / 16

        kernel_Gaussian_Laplacian = torch.zeros(1, 1, 5, 5).cuda()
        kernel_Gaussian_Laplacian[:, :, 0, 2] = a3
        kernel_Gaussian_Laplacian[:, :, 1, 1] = a3
        kernel_Gaussian_Laplacian[:, :, 1, 2] = a2
        kernel_Gaussian_Laplacian[:, :, 1, 3] = a3
        kernel_Gaussian_Laplacian[:, :, 2, 0] = a3
        kernel_Gaussian_Laplacian[:, :, 2, 1] = a2
        kernel_Gaussian_Laplacian[:, :, 2, 2] = a1
        kernel_Gaussian_Laplacian[:, :, 2, 3] = a2
        kernel_Gaussian_Laplacian[:, :, 2, 4] = a3
        kernel_Gaussian_Laplacian[:, :, 3, 1] = a3
        kernel_Gaussian_Laplacian[:, :, 3, 2] = a2
        kernel_Gaussian_Laplacian[:, :, 3, 3] = a3
        kernel_Gaussian_Laplacian[:, :, 4, 2] = a3

        x_GL = F.conv2d(image_a, kernel_Gaussian_Laplacian, stride=1, padding=2)
        threshold = 1e-6
        x_GL=torch.abs(x_GL)
        x_GL[(torch.abs(x_GL) < threshold)] = 0
        x_GL = torch.nn.functional.relu(x_GL)

        b, c, h, w = x_GL.size()
        x_GL_t = x_GL.view(b, c, -1)
        x_GL_max = torch.max(x_GL_t, dim=-1)[0]
        all_sum = torch.sum((x_GL_t > 0), dim=-1)
        ###########统计_sub_sum
        x_values = []
        index_values = []
        start = 0
        end = 10
        steps = 100
        for i in range(steps + 1):
            xx = (end - start) * (i / steps) ** 3 + start
            x_values.append(xx)
            index_values.append(i)
        x_values = torch.tensor(x_values)
        index_values = torch.tensor(index_values)

        x_values = x_values.unsqueeze(0).unsqueeze(2).cuda()
        sub_sum = ((x_GL_t > 0) & (x_GL_t <= x_values))
        sub_sum = torch.sum(sub_sum, dim=-1, dtype=torch.int)  # 加入dtype=torch.int可以提高速度
        percentage = sub_sum / all_sum

        p1 = 0.2+0.05*weight_att[:,1]
        p2 = 0.4+0.05*weight_att[:,2]
        p3 = 0.6+0.05*weight_att[:,3]
        p4 = 0.8+0.05*weight_att[:,4]
        parameters = [p1, p2, p3, p4]
        parameters= torch.stack(parameters)
        sorted_parameters, _ = torch.sort(parameters, dim=0)
        threshold_percentage=sorted_parameters.transpose(0,1)
        percentage = percentage.unsqueeze(1)
        threshold_percentage = threshold_percentage.unsqueeze(2)

        z = torch.abs(percentage - threshold_percentage)
        closest_positions = torch.argmin(z, dim=-1)
        threshold = x_values[:, closest_positions].squeeze(0).squeeze(-1)
        x_GL_max = x_GL_max
        x_GL_min = torch.zeros_like(x_GL_max)
        threshold_min = torch.cat((x_GL_min, threshold), dim=-1)
        threshold_max = torch.cat((threshold, x_GL_max), dim=-1)

        threshold_min = threshold_min.unsqueeze(2)
        threshold_max = threshold_max.unsqueeze(2)
        mask = ((x_GL_t > threshold_min) & (x_GL_t <= threshold_max))
        x_GL_mask = mask.float()
        x_GL_mask = x_GL_mask.view(b, 5, h, w)
        kernel_size = (9, 9)
        stride = 2
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        average_pooling = torch.nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        average_pooling2 = torch.nn.AvgPool2d(kernel_size, stride=2, padding=padding)


        x_GL_mask_mean = average_pooling(x_GL_mask)
        x_GL_mask_mean = average_pooling(x_GL_mask_mean)
        x_GL_mask_mean = average_pooling2(x_GL_mask_mean)

        x_GL_mask_mean = x_GL_mask_mean.view(b, 5, -1)
        min_values, _ = torch.min(x_GL_mask_mean, dim=2, keepdim=True)
        max_values, _ = torch.max(x_GL_mask_mean, dim=2, keepdim=True)

        normalized_image = (x_GL_mask_mean - min_values) / (max_values - min_values)
        normalized_image = normalized_image.view(b, 5, h // 2, w // 2)

        return normalized_image


class backbone_HGE_HGA(nn.Module):
    def __init__(self,
                 backbone_name='resnet18',
                 pretrained=True,
                 # pretrained=False,
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True,pretrained=pretrained)
        self.ac1 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.act1,
        )
        self.maxpool = self.backbone.maxpool
#Gradient Decoupling Masks Generation
        self.gdm = GDM()
#Hybrid Gradient Enhancement
        self.hge1 = HGE(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.hge2 = HGE(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.hge3 = HGE(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(256)
        self.hge4 = HGE(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4=nn.BatchNorm2d(512)
# Stage1
        self.layer1_0_conv1 = self.backbone.layer1[0].conv1
        self.layer1_0_bn1 = self.backbone.layer1[0].bn1
        self.layer1_0_drop_block = self.backbone.layer1[0].drop_block
        self.layer1_0_act1 = self.backbone.layer1[0].act1
        self.layer1_0_aa = self.backbone.layer1[0].aa
        self.layer1_0_conv2 = self.backbone.layer1[0].conv2
        self.layer1_0_bn2 = self.backbone.layer1[0].bn2
        self.layer1_0_act2 = self.backbone.layer1[0].act2
        self.layer1_1_conv1 = self.backbone.layer1[1].conv1
        self.layer1_1_bn1 = self.backbone.layer1[1].bn1
        self.layer1_1_drop_block = self.backbone.layer1[1].drop_block
        self.layer1_1_act1 = self.backbone.layer1[1].act1
        self.layer1_1_aa = self.backbone.layer1[1].aa
        self.layer1_1_conv2 = self.backbone.layer1[1].conv2
        self.layer1_1_bn2 = self.backbone.layer1[1].bn2
        self.layer1_1_act2 = self.backbone.layer1[1].act2
#stage2
        self.layer2_0_conv1 = self.backbone.layer2[0].conv1
        self.layer2_0_bn1 = self.backbone.layer2[0].bn1
        self.layer2_0_drop_block = self.backbone.layer2[0].drop_block
        self.layer2_0_act1 = self.backbone.layer2[0].act1
        self.layer2_0_aa = self.backbone.layer2[0].aa
        self.layer2_0_conv2 = self.backbone.layer2[0].conv2
        self.layer2_0_bn2 = self.backbone.layer2[0].bn2
        self.layer2_0_act2 = self.backbone.layer2[0].act2
        self.layer2_0_downsample=self.backbone.layer2[0].downsample
        self.layer2_1_conv1 = self.backbone.layer2[1].conv1
        self.layer2_1_bn1 = self.backbone.layer2[1].bn1
        self.layer2_1_drop_block = self.backbone.layer2[1].drop_block
        self.layer2_1_act1 = self.backbone.layer2[1].act1
        self.layer2_1_aa = self.backbone.layer2[1].aa
        self.layer2_1_conv2 = self.backbone.layer2[1].conv2
        self.layer2_1_bn2 = self.backbone.layer2[1].bn2
        self.layer2_1_act2 = self.backbone.layer2[1].act2

# stage3
        self.layer3_0_conv1 = self.backbone.layer3[0].conv1
        self.layer3_0_bn1 = self.backbone.layer3[0].bn1
        self.layer3_0_drop_block = self.backbone.layer3[0].drop_block
        self.layer3_0_act1 = self.backbone.layer3[0].act1
        self.layer3_0_aa = self.backbone.layer3[0].aa
        self.layer3_0_conv2 = self.backbone.layer3[0].conv2
        self.layer3_0_bn2 = self.backbone.layer3[0].bn2
        self.layer3_0_act2 = self.backbone.layer3[0].act2
        self.layer3_0_downsample = self.backbone.layer3[0].downsample
        self.layer3_1_conv1 = self.backbone.layer3[1].conv1
        self.layer3_1_bn1 = self.backbone.layer3[1].bn1
        self.layer3_1_drop_block = self.backbone.layer3[1].drop_block
        self.layer3_1_act1 = self.backbone.layer3[1].act1
        self.layer3_1_aa = self.backbone.layer3[1].aa
        self.layer3_1_conv2 = self.backbone.layer3[1].conv2
        self.layer3_1_bn2 = self.backbone.layer3[1].bn2
        self.layer3_1_act2 = self.backbone.layer3[1].act2
# stage4
        self.layer4_0_conv1 = self.backbone.layer4[0].conv1
        self.layer4_0_bn1 = self.backbone.layer4[0].bn1
        self.layer4_0_drop_block = self.backbone.layer4[0].drop_block
        self.layer4_0_act1 = self.backbone.layer4[0].act1
        self.layer4_0_aa = self.backbone.layer4[0].aa
        self.layer4_0_conv2 = self.backbone.layer4[0].conv2
        self.layer4_0_bn2 = self.backbone.layer4[0].bn2
        self.layer4_0_act2 = self.backbone.layer4[0].act2
        self.layer4_0_downsample = self.backbone.layer4[0].downsample
        self.layer4_1_conv1 = self.backbone.layer4[1].conv1
        self.layer4_1_bn1 = self.backbone.layer4[1].bn1
        self.layer4_1_drop_block = self.backbone.layer4[1].drop_block
        self.layer4_1_act1 = self.backbone.layer4[1].act1
        self.layer4_1_aa = self.backbone.layer4[1].aa
        self.layer4_1_conv2 = self.backbone.layer4[1].conv2
        self.layer4_1_bn2 = self.backbone.layer4[1].bn2
        self.layer4_1_act2 = self.backbone.layer4[1].act2

        self.act=nn.ReLU()

        self.fc1 = nn.Linear(64,5)
        self.fc2 = nn.Linear(128,5)
        self.fc3 = nn.Linear(256,5)
        self.fc4 = nn.Linear(512,5)


        self.sigmoid =nn.Sigmoid()


    def forward(self, x):

        out = []
        b,c, h, w = x.size()
#Gradient Decoupling Masks generation
        GDM1 = self.gdm(x).cuda()
        GDM2 = F.max_pool2d(GDM1, kernel_size=2, stride=2)  ##池化
        GDM3 = F.max_pool2d(GDM2, kernel_size=2, stride=2)  ##池化
        GDM4 = F.max_pool2d(GDM3, kernel_size=2, stride=2)  ##池化

        x = self.ac1(x)
        x = self.maxpool(x)
        #stage1
        #basicblock
        shortcut = x
        x = self.layer1_0_conv1(x)
        x = self.layer1_0_bn1(x)
        x = self.layer1_0_drop_block(x)
        x = self.layer1_0_act1(x)
        x = self.layer1_0_aa(x)
        x = self.layer1_0_conv2(x)
        x = self.layer1_0_bn2(x)
        x += shortcut
        x = self.layer1_0_act2(x)
        # basicblock
        shortcut = x
        x = self.layer1_1_conv1(x)
        x = self.layer1_1_bn1(x)
        x = self.layer1_1_drop_block(x)
        x = self.layer1_1_act1(x)
        x = self.layer1_1_aa(x)
        x = self.layer1_1_conv2(x)
        x = self.layer1_1_bn2(x)
        x += shortcut
        x = self.layer1_1_act2(x)

        # HGA #Learn the weights of the sub-masks GDSM of the gradient decoupling mask GDM
        weight_att = torch.mean(x,dim=(2,3),keepdim=True)
        weight_att = torch.squeeze(weight_att,dim=2)
        weight_att = torch.squeeze(weight_att,dim=2)
        weight_att = self.fc1(weight_att)
        weight_att = torch.softmax(weight_att,dim=1)

        x_sob = self.hge1(x)
        x_sob = self.bn1(x_sob)
        x_sob = self.act(x_sob)
        # out_feature1=x_sob  ######feature
        x_out=x+x_sob
        x_out = self.act(x_out)
        ###################boundary_att
        GDSM_att1=GDM1*weight_att.view(b,5,1,1)
        split_tensors1 = torch.split(GDSM_att1, 1, dim=1)
        x_att = x_out * split_tensors1[0] + x_out * split_tensors1[1]+ x_out * split_tensors1[2]+ x_out * split_tensors1[3]+ x_out * split_tensors1[4]
        x_out = x_out + x_att
        x_out = self.act(x_out)
        ###################
        out.append(x_out)
        #layer2
        #basicblock
        shortcut = x
        x = self.layer2_0_conv1(x)
        x = self.layer2_0_bn1(x)
        x = self.layer2_0_drop_block(x)
        x = self.layer2_0_act1(x)
        x = self.layer2_0_aa(x)
        x = self.layer2_0_conv2(x)
        x = self.layer2_0_bn2(x)
        shortcut = self.layer2_0_downsample(shortcut)
        x += shortcut
        x = self.layer2_0_act2(x)
        #basicblock
        shortcut = x
        x = self.layer2_1_conv1(x)
        x = self.layer2_1_bn1(x)
        x = self.layer2_1_drop_block(x)
        x = self.layer2_1_act1(x)
        x = self.layer2_1_aa(x)
        x = self.layer2_1_conv2(x)
        x = self.layer2_1_bn2(x)
        x += shortcut
        x = self.layer2_1_act2(x)

        ###################weight_att
        weight_att = torch.mean(x,dim=(2,3),keepdim=True)
        weight_att = torch.squeeze(weight_att,dim=2)
        weight_att = torch.squeeze(weight_att,dim=2)
        weight_att = self.fc2(weight_att)
        weight_att = torch.softmax(weight_att,dim=1)

        x_sob = self.hge2(x)
        x_sob = self.bn2(x_sob)
        x_sob = self.act(x_sob)
        # out_feature2=x_sob ######feature
        x_out=x+x_sob
        x_out = self.act(x_out)
        ###################boundary_ztt
        GDSM_att2=GDM2*weight_att.view(b,5,1,1)
        split_tensors2 = torch.split(GDSM_att2, 1, dim=1)
        x_att = x_out * split_tensors2[0] + x_out * split_tensors2[1]+ x_out * split_tensors2[2]+ x_out * split_tensors2[3]+ x_out * split_tensors2[4]
        x_out = x_out + x_att
        x_out = self.act(x_out)
        ###################
        out.append(x_out)
        # layer3
        # basicblock
        shortcut = x
        x = self.layer3_0_conv1(x)
        x = self.layer3_0_bn1(x)
        x = self.layer3_0_drop_block(x)
        x = self.layer3_0_act1(x)
        x = self.layer3_0_aa(x)
        x = self.layer3_0_conv2(x)
        x = self.layer3_0_bn2(x)
        shortcut = self.layer3_0_downsample(shortcut)
        x += shortcut
        x = self.layer3_0_act2(x)
        # basicblock
        shortcut = x
        x = self.layer3_1_conv1(x)
        x = self.layer3_1_bn1(x)
        x = self.layer3_1_drop_block(x)
        x = self.layer3_1_act1(x)
        x = self.layer3_1_aa(x)
        x = self.layer3_1_conv2(x)
        x = self.layer3_1_bn2(x)
        x += shortcut
        x = self.layer3_1_act2(x)
        ###################weight_att
        weight_att = torch.mean(x, dim=(2, 3), keepdim=True)
        weight_att = torch.squeeze(weight_att, dim=2)
        weight_att = torch.squeeze(weight_att, dim=2)
        weight_att = self.fc3(weight_att)
        weight_att = torch.softmax(weight_att, dim=1)

        x_sob = self.hge3(x)
        x_sob = self.bn3(x_sob)
        x_sob = self.act(x_sob)
        # out_feature3=x_sob ######feature
        x_out=x+x_sob
        x_out = self.act(x_out)
        ###################boundary_ztt
        GDSM_att3 = GDM3 * weight_att.view(b, 5, 1, 1)
        split_tensors3 = torch.split(GDSM_att3, 1, dim=1)
        x_att = x_out * split_tensors3[0] + x_out * split_tensors3[1] + x_out * split_tensors3[2] + x_out * split_tensors3[3] + x_out * \
                split_tensors3[4]
        x_out = x_out + x_att
        x_out = self.act(x_out)
        ###################
        out.append(x_out)

        # layer4
        # basicblock
        shortcut = x
        x = self.layer4_0_conv1(x)
        x = self.layer4_0_bn1(x)
        x = self.layer4_0_drop_block(x)
        x = self.layer4_0_act1(x)
        x = self.layer4_0_aa(x)
        x = self.layer4_0_conv2(x)
        x = self.layer4_0_bn2(x)
        shortcut = self.layer4_0_downsample(shortcut)
        x += shortcut
        x = self.layer4_0_act2(x)
        # basicblock
        shortcut = x
        x = self.layer4_1_conv1(x)
        x = self.layer4_1_bn1(x)
        x = self.layer4_1_drop_block(x)
        x = self.layer4_1_act1(x)
        x = self.layer4_1_aa(x)
        x = self.layer4_1_conv2(x)
        x = self.layer4_1_bn2(x)
        x += shortcut
        x = self.layer4_1_act2(x)
        ###################weight_att
        weight_att = torch.mean(x, dim=(2, 3), keepdim=True)
        weight_att = torch.squeeze(weight_att, dim=2)
        weight_att = torch.squeeze(weight_att, dim=2)
        weight_att = self.fc4(weight_att)
        weight_att = torch.softmax(weight_att, dim=1)

        x_sob = self.hge4(x)
        x_sob = self.bn4(x_sob)
        x_sob = self.act(x_sob)
        # out_feature4=x_sob ######feature
        x_out=x+x_sob
        x_out = self.act(x_out)
        ###################boundary_ztt
        GDSM_att4 = GDM4 * weight_att.view(b, 5, 1, 1)
        split_tensors4 = torch.split(GDSM_att4, 1, dim=1)
        x_att = x_out * split_tensors4[0] + x_out * split_tensors4[1] + x_out * split_tensors4[2] + x_out * split_tensors4[3] + x_out * \
                split_tensors4[4]
        x_out = x_out + x_att
        x_out = self.act(x_out)
        # ###################
        out.append(x_out)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=128,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)


        self.b1 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p1 = WF(encoder_channels[-4], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        # self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)
            x = self.b1(x)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)
            x = self.b1(x)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class GDGNet(nn.Module):
    def __init__(self,
                 decode_channels=128,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()
        self.backbone = backbone_HGE_HGA()

        encoder_channels = [64, 128, 256, 512]

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

import albumentations as albu
def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

from PIL import Image
if __name__ == "__main__":
    # x = torch.randn(8, 3, 512, 512)

    image_1 = Image.open(r"F:\Semantic-segmentation\GeoSeg-main\data\vaihingen4\test\images_1024\top_mosaic_09cm_area10_0_0.tif").convert('RGB')
    img = np.array(image_1)
    aug = get_val_transform()(image=img.copy())
    img = aug['image']
    image_1 = torch.from_numpy(img).permute(2, 0, 1).float()

    image_2 = Image.open(r"F:\Semantic-segmentation\GeoSeg-main\data\vaihingen4\test\images_1024\top_mosaic_09cm_area10_0_1.tif").convert('RGB')
    img = np.array(image_2)
    aug = get_val_transform()(image=img.copy())
    img = aug['image']
    image_2 = torch.from_numpy(img).permute(2, 0, 1).float()
    x = torch.cat((image_1.unsqueeze(0), image_2.unsqueeze(0)), dim=0)

    net1 = GDGNet(num_classes=6)

    for name, module in net1.decoder.named_children():
        num_params = count_parameters(module)
        print(f"{name}: {num_params} parameters")
    print(f"Total parameters in the model: {count_parameters(net1)}")
    net1=net1.cuda()
    x=x.cuda()

    out11 = net1(x)
    print(out11)
