# import torch
# from torch import nn as nn
# from torch.nn import functional as F

# from basicsr.models.archs.arch_util import default_init_weights, make_layer


# class ResidualDenseBlock(nn.Module):
#     """Residual Dense Block.

#     Used in RRDB block in ESRGAN.

#     Args:
#         num_feat (int): Channel number of intermediate features.
#         num_grow_ch (int): Channels for each growth.
#     """

#     def __init__(self, num_feat=64, num_grow_ch=32):
#         super(ResidualDenseBlock, self).__init__()
#         self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
#         self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
#         self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1,
#                                1)
#         self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1,
#                                1)
#         self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         # initialization
#         default_init_weights(
#             [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         # Emperically, we use 0.2 to scale the residual for better performance
#         return x5 * 0.2 + x


# class RRDB(nn.Module):
#     """Residual in Residual Dense Block.

#     Used in RRDB-Net in ESRGAN.

#     Args:
#         num_feat (int): Channel number of intermediate features.
#         num_grow_ch (int): Channels for each growth.
#     """

#     def __init__(self, num_feat, num_grow_ch=32):
#         super(RRDB, self).__init__()
#         self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
#         self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
#         self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

#     def forward(self, x):
#         out = self.rdb1(x)
#         out = self.rdb2(out)
#         out = self.rdb3(out)
#         # Emperically, we use 0.2 to scale the residual for better performance
#         return out * 0.2 + x


# class RRDBNet(nn.Module):
#     """Networks consisting of Residual in Residual Dense Block, which is used
#     in ESRGAN.

#     ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
#     Currently, it supports x4 upsampling scale factor.

#     Args:
#         num_in_ch (int): Channel number of inputs.
#         num_out_ch (int): Channel number of outputs.
#         num_feat (int): Channel number of intermediate features.
#             Default: 64
#         num_block (int): Block number in the trunk network. Defaults: 23
#         num_grow_ch (int): Channels for each growth. Default: 32.
#     """

#     def __init__(self,
#                  num_in_ch,
#                  num_out_ch,
#                  num_feat=64,
#                  num_block=23,
#                  num_grow_ch=32):
#         super(RRDBNet, self).__init__()
#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body = make_layer(
#             RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
#         self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         # upsample
#         self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         feat = self.conv_first(x)
#         body_feat = self.conv_body(self.body(feat))
#         feat = feat + body_feat
#         # upsample
#         feat = self.lrelu(
#             self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
#         feat = self.lrelu(
#             self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
#         out = self.conv_last(self.lrelu(self.conv_hr(feat)))
#         return out
# import torch 
# from torch import nn as nn
# from torch.nn import functional as F 
# from basicsr.models.archs.arch_util import default_init_weights, make_layer

# class ESAblock(nn.Module):
#     def __init__(self,numfeat=64,halfeat = 16):
#         super(ESAblock,self).__init__()
#         self.conv1 = nn.Conv2d(numfeat,halfeat,kernel_size=1,padding=0)
#         self.conv2 = nn.Conv2d(halfeat,halfeat,kernel_size=1,padding=0)
#         self.conv3 = nn.Conv2d(halfeat,halfeat,kernel_size=3,padding=1)
#         self.conv4 = nn.Conv2d(halfeat,halfeat,kernel_size=3,padding=0,stride=2)
#         self.conv5 = nn.Conv2d(halfeat,halfeat,kernel_size=3,padding=1)
#         self.conv6 = nn.Conv2d(halfeat,numfeat,kernel_size=3,padding=1)
#         self.sigmod = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self,x):
#         #print(x.size())
#         x1 = self.conv1(x)
#         x2 = self.conv4(x1)
#         #print(x2.size())
#         x3 = F.max_pool2d(x2,kernel_size=7,stride=3)
#         #print(x3.size())
#         x4 = self.relu(self.conv3(x3))
#         #print(x4.size())
#         x5 = self.relu(self.conv5(x4))
#         x5 = F.interpolate(x5,(x.size(2),x.size(3)),mode= 'bilinear',align_corners=False)
#         #print(x5.size())
#         x6 = (self.conv2(x1))
#         x7 = self.conv6(x5+x6)
#         m = self.sigmod(x7)
#         out = x*m
#         return out
# class ResidualDenseBlock(nn.Module):
#     """Residual Dense Block.

#     Used in RRDB block in ESRGAN.

#     Args:
#         num_feat (int): Channel number of intermediate features.
#         num_grow_ch (int): Channels for each growth.
#     """

#     def __init__(self, num_feat=64, num_grow_ch=32):
#         super(ResidualDenseBlock, self).__init__()
#         self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
#         self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
#         self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1,
#                                1)
#         self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1,
#                                1)
#         self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         # initialization
#         default_init_weights(
#             [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         # Emperically, we use 0.2 to scale the residual for better performance
#         return x5 * 0.2 + x
# class RRDB(nn.Module):
#     """Residual in Residual Dense Block.

#     Used in RRDB-Net in ESRGAN.

#     Args:
#         num_feat (int): Channel number of intermediate features.
#         num_grow_ch (int): Channels for each growth.
#     """

#     def __init__(self, num_feat, num_grow_ch=32):
#         super(RRDB, self).__init__()
#         self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
#         self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
#         self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
#         #self.eca0 = ESAblock(num_feat,16)
#         self.eca1 = ESAblock(num_feat,16)
#         self.eca2 = ESAblock(num_feat,16)
#         self.eca3 = ESAblock(num_feat,16)
#         self.cat =nn.Conv2d(6*num_feat,num_feat,kernel_size=3,padding=1)

#     def forward(self, x):
#         #out00 = self.eca0(x)
#         out1= self.rdb1(x)
#         out11 = self.eca1(out1)
#         out2 = self.rdb2(out11)
#         out22 = self.eca2(out2)
#         out3 = self.rdb3(out22)
#         out33 = self.eca3(out3)
#         out = self.cat(torch.cat((out1,out11,out2,out22,out3,out33),1))
#         # Emperically, we use 0.2 to scale the residual for better performance
#         return out * 0.2 + x
# class RRDBNet(nn.Module):
#     """Networks consisting of Residual in Residual Dense Block, which is used
#     in ESRGAN.

#     ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
#     Currently, it supports x4 upsampling scale factor.

#     Args:
#         num_in_ch (int): Channel number of inputs.
#         num_out_ch (int): Channel number of outputs.
#         num_feat (int): Channel number of intermediate features.
#             Default: 64
#         num_block (int): Block number in the trunk network. Defaults: 23
#         num_grow_ch (int): Channels for each growth. Default: 32.
#     """

#     def __init__(self,
#                  num_in_ch,
#                  num_out_ch,
#                  num_feat=64,
#                  num_block=23,
#                  num_grow_ch=32):
#         super(RRDBNet, self).__init__()
#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body = make_layer(
#             RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
#         self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         # upsample
#         self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         feat = self.conv_first(x)
#         body_feat = self.conv_body(self.body(feat))
#         feat = feat + body_feat
#         # upsample
#         feat = self.lrelu(
#             self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
#         feat = self.lrelu(
#             self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
#         out = self.conv_last(self.lrelu(self.conv_hr(feat)))
#         return out



import torch 
from torch import nn as nn
from torch.nn import functional as F 
from basicsr.models.archs.arch_util import default_init_weights, make_layer

class ESAblock(nn.Module):
    def __init__(self,numfeat=64,halfeat = 16):
        super(ESAblock,self).__init__()
        self.conv1 = nn.Conv2d(numfeat,halfeat,kernel_size=1,padding=0)
        self.conv2 = nn.Conv2d(halfeat,halfeat,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(halfeat,halfeat,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(halfeat,halfeat,kernel_size=3,padding=0,stride=2)
        self.conv5 = nn.Conv2d(halfeat,halfeat,kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(halfeat,numfeat,kernel_size=3,padding=1)
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        #print(x.size())
        x1 = self.conv1(x)
        x2 = self.conv4(x1)
        #print(x2.size())
        x3 = F.max_pool2d(x2,kernel_size=7,stride=3)
        #print(x3.size())
        x4 = self.relu(self.conv3(x3))
        #print(x4.size())
        x5 = self.relu(self.conv5(x4))
        x5 = F.interpolate(x5,(x.size(2),x.size(3)),mode= 'bilinear',align_corners=False)
        #print(x5.size())
        x6 = (self.conv2(x1))
        x7 = self.conv6(x5+x6)
        m = self.sigmod(x7)
        out = x*m
        return out
class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=64):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
       # self.esa1 = ESAblock(num_grow_ch,16)
        #self.esa2 = ESAblock(num_grow_ch,16)
        #self.esa3 = ESAblock(num_grow_ch,16)
        #self.esa4 = ESAblock(num_grow_ch,16)
        self.esa5 = ESAblock(num_feat,16)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1,1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1,1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        #x1 = self.esa1(x1)
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        #x2 = self.esa2(x2)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        #x3 = self.esa3(x3)
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        #x4 = self.esa4(x4)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.esa5(x5)
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x
class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.esa1 = ESAblock(num_feat,16)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.esa2 = ESAblock(num_feat,16)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.esa3 = ESAblock(num_feat,16)
        self.cat =nn.Conv2d(6*num_feat,num_feat,kernel_size=3,padding=1)

    def forward(self, x):
        out1= self.rdb1(x)
        out11 = self.esa1(out1)
        out2 = self.rdb2(out11)
        out22 = self.esa2(out2)
        out3 = self.rdb3(out22)
        out33 = self.esa3(out3)
        
        out = self.cat(torch.cat((out1,out11,out2,out22,out3,out33),1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
