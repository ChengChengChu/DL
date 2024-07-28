# import torch
# from torch import nn
# from torch.nn.functional import relu

# class UNet(nn.Module):
#     def __init__(self, n_class):
#         super().__init__()

#         ###### encoder ######
#         self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
#         self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
#         self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 
#         self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

#         self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) 
#         self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) 

#         ##### decoder #######
#         self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
#         self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

#         self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

#         self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
#         self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

#         self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

#         # Output layer
#         self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
    
#     def forward(self, x) :
#         # x.size: 3x256x256
#         ######## unet encoder #######
#         xe11 = self.e11(x) 
#         xe11 = relu(xe11) 
#         xe12 = self.e12(xe11) 
#         xe12 = relu(xe12) 
#         xp1 = self.pool1(xe12) 
        
#         xe21 = self.e21(xp1)
#         xe21 = relu(xe21) 
#         xe22 = self.e22(xe21)
#         xe22 = relu(xe22) 
#         xp2 = self.pool2(xe22) 


#         xe31 = self.e31(xp2)
#         xe31 = relu(xe31) 
#         xe32 = self.e32(xe31)
#         xe32 = relu(xe32)
#         xp3 = self.pool3(xe32) 

#         xe41 = self.e41(xp3)
#         xe41 = relu(xe41)
#         xe42 = self.e42(xe41)
#         xe42 = relu(xe42)
#         xp4 = self.pool4(xe42) 

#         xe51 = relu(self.e51(xp4))
#         xe52 = relu(self.e52(xe51)) # xe52.size: 1024 x 8 x 8

#         print(f"layer1 : {xp1.size()}\nlayer2 : {xp2.size()}\nlayer3 : {xp3.size()}\nlayer4 : {xp4.size()}\nlayer5 : {xe52.size()}")

#         import pdb
#         pdb.set_trace()

#         xu1 = self.upconv1(xe52)
#         xu11 = torch.cat([xu1, xe42], dim=1)
#         xd11 = relu(self.d11(xu11))
#         xd12 = relu(self.d12(xd11))


#         xu2 = self.upconv2(xd12)
#         xu22 = torch.cat([xu2, xe32], dim=1)
#         xd21 = relu(self.d21(xu22))
#         xd22 = relu(self.d22(xd21))

#         xu3 = self.upconv3(xd22)
#         xu33 = torch.cat([xu3, xe22], dim=1)
#         xd31 = relu(self.d31(xu33))
#         xd32 = relu(self.d32(xd31))

#         xu4 = self.upconv4(xd32)
#         xu44 = torch.cat([xu4, xe12], dim=1)
#         xd41 = relu(self.d41(xu44))
#         xd42 = relu(self.d42(xd41))

#         # Output layer
#         out = self.outconv(xd42)

        
        
       
        


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
       
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

        

       

# from oxford_pet import *

# dataset = load_dataset('../../dataset/oxford-iiit-pet/', mode='valid')
# model = UNet(n_classes=2, n_channels=3)
# test_loader = DataLoader(dataset, batch_size=1)

# for x in test_loader :
#     input_img = x['image']
#     input_img = input_img.to(torch.float32)
#     # import pdb
#     # pdb.set_trace()
#     output = model(input_img)
    
    # print(type(input_img))
    # break