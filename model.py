import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    '''(Conv3d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(ConvBlock3D, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class UNet3D_3(nn.Module):
    def __init__(self, base=32):
        super(UNet3D_3, self).__init__()

        self.base = base

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Define the network layers using the base size
        self.enc_conv1 = ConvBlock3D(in_ch=1, out_ch=self.base)
        self.enc_conv2 = ConvBlock3D(in_ch=self.base, out_ch=2*self.base)
        self.enc_conv3 = ConvBlock3D(in_ch=2*self.base, out_ch=4*self.base)

        self.conv_b = ConvBlock3D(in_ch=4*self.base, out_ch=8*self.base)

        self.tconv3 = nn.ConvTranspose3d(in_channels=8*self.base, out_channels=4*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv3 = ConvBlock3D(in_ch=8*self.base, out_ch=4*self.base)
        self.tconv2 = nn.ConvTranspose3d(in_channels=4*self.base, out_channels=2*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv2 = ConvBlock3D(in_ch=4*self.base, out_ch=2*self.base)
        self.tconv1 = nn.ConvTranspose3d(in_channels=2*self.base, out_channels=self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv1 = ConvBlock3D(in_ch=2*self.base, out_ch=self.base)

        self.outconv = nn.Conv3d(in_channels=self.base, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Forward pass through the UNet model
        enc_x1 = self.enc_conv1(x)
        x_p = self.pool(enc_x1)
        enc_x2 = self.enc_conv2(x_p)
        x_p = self.pool(enc_x2)
        enc_x3 = self.enc_conv3(x_p)
        x_p = self.pool(enc_x3)

        x_c = self.conv_b(x_p)

        x3_t = self.tconv3(x_c)
        x_c = torch.cat([enc_x3, x3_t], dim=1)
        dec_x3 = self.dec_conv3(x_c)
        x2_t = self.tconv2(dec_x3)
        x_c = torch.cat([enc_x2, x2_t], dim=1)
        dec_x2 = self.dec_conv2(x_c)
        x1_t = self.tconv1(dec_x2)
        x_c = torch.cat([enc_x1, x1_t], dim=1)
        dec_x1 = self.dec_conv1(x_c)

        x_final = self.outconv(dec_x1)

        return x_final

    

class UNet3D_4(nn.Module):
    def __init__(self, base=32):
        super(UNet3D_4, self).__init__()

        self.base = base

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Define the network layers using the base size
        self.enc_conv1 = ConvBlock3D(in_ch=1, out_ch=self.base)
        self.enc_conv2 = ConvBlock3D(in_ch=self.base, out_ch=2*self.base)
        self.enc_conv3 = ConvBlock3D(in_ch=2*self.base, out_ch=4*self.base)     
        self.enc_conv4 = ConvBlock3D(in_ch=4*self.base, out_ch=8*self.base)

        self.conv_b = ConvBlock3D(in_ch=8*self.base, out_ch=16*self.base)

        self.tconv4 = nn.ConvTranspose3d(in_channels=16*self.base, out_channels=8*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv4 = ConvBlock3D(in_ch=16*self.base, out_ch=8*self.base)
        self.tconv3 = nn.ConvTranspose3d(in_channels=8*self.base, out_channels=4*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv3 = ConvBlock3D(in_ch=8*self.base, out_ch=4*self.base)
        self.tconv2 = nn.ConvTranspose3d(in_channels=4*self.base, out_channels=2*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv2 = ConvBlock3D(in_ch=4*self.base, out_ch=2*self.base)
        self.tconv1 = nn.ConvTranspose3d(in_channels=2*self.base, out_channels=self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv1 = ConvBlock3D(in_ch=2*self.base, out_ch=self.base)

        self.outconv = nn.Conv3d(in_channels=self.base, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Forward pass through the UNet model
        enc_x1 = self.enc_conv1(x)
        x_p = self.pool(enc_x1)
        enc_x2 = self.enc_conv2(x_p)
        x_p = self.pool(enc_x2)
        enc_x3 = self.enc_conv3(x_p)
        x_p = self.pool(enc_x3)
        enc_x4 = self.enc_conv4(x_p)
        x_p = self.pool(enc_x4)

        x_c = self.conv_b(x_p)

        x4_t = self.tconv4(x_c)
        x_c = torch.cat([enc_x4, x4_t], dim=1)
        dec_x4 = self.dec_conv4(x_c)
        x3_t = self.tconv3(dec_x4)
        x_c = torch.cat([enc_x3, x3_t], dim=1)
        dec_x3 = self.dec_conv3(x_c)
        x2_t = self.tconv2(dec_x3)
        x_c = torch.cat([enc_x2, x2_t], dim=1)
        dec_x2 = self.dec_conv2(x_c)
        x1_t = self.tconv1(dec_x2)
        x_c = torch.cat([enc_x1, x1_t], dim=1)
        dec_x1 = self.dec_conv1(x_c)

        x_final = self.outconv(dec_x1)

        return x_final



class UNet3D_5(nn.Module):
    def __init__(self, base=32):
        super(UNet3D_5, self).__init__()

        self.base = base

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Define the network layers using the base size
        self.enc_conv1 = ConvBlock3D(in_ch=1, out_ch=self.base)
        self.enc_conv2 = ConvBlock3D(in_ch=self.base, out_ch=2*self.base)
        self.enc_conv3 = ConvBlock3D(in_ch=2*self.base, out_ch=4*self.base)     
        self.enc_conv4 = ConvBlock3D(in_ch=4*self.base, out_ch=8*self.base)
        self.enc_conv5 = ConvBlock3D(in_ch=8*self.base, out_ch=16*self.base)

        self.conv_b = ConvBlock3D(in_ch=16*self.base, out_ch=32*self.base)

        self.tconv5 = nn.ConvTranspose3d(in_channels=32*self.base, out_channels=16*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv5 = ConvBlock3D(in_ch=32*self.base, out_ch=16*self.base)
        self.tconv4 = nn.ConvTranspose3d(in_channels=16*self.base, out_channels=8*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv4 = ConvBlock3D(in_ch=16*self.base, out_ch=8*self.base)
        self.tconv3 = nn.ConvTranspose3d(in_channels=8*self.base, out_channels=4*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv3 = ConvBlock3D(in_ch=8*self.base, out_ch=4*self.base)
        self.tconv2 = nn.ConvTranspose3d(in_channels=4*self.base, out_channels=2*self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv2 = ConvBlock3D(in_ch=4*self.base, out_ch=2*self.base)
        self.tconv1 = nn.ConvTranspose3d(in_channels=2*self.base, out_channels=self.base, kernel_size=2, stride=2, padding=0)
        self.dec_conv1 = ConvBlock3D(in_ch=2*self.base, out_ch=self.base)

        self.outconv = nn.Conv3d(in_channels=self.base, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Forward pass through the UNet model
        enc_x1 = self.enc_conv1(x)
        x_p = self.pool(enc_x1)
        enc_x2 = self.enc_conv2(x_p)
        x_p = self.pool(enc_x2)
        enc_x3 = self.enc_conv3(x_p)
        x_p = self.pool(enc_x3)
        enc_x4 = self.enc_conv4(x_p)
        x_p = self.pool(enc_x4)
        enc_x5 = self.enc_conv5(x_p)
        x_p = self.pool(enc_x5)

        x_c = self.conv_b(x_p)

        x5_t = self.tconv5(x_c)
        x_c = torch.cat([enc_x5, x5_t], dim=1)
        dec_x5 = self.dec_conv5(x_c)
        x4_t = self.tconv4(dec_x5)
        x_c = torch.cat([enc_x4, x4_t], dim=1)
        dec_x4 = self.dec_conv4(x_c)
        x3_t = self.tconv3(dec_x4)
        x_c = torch.cat([enc_x3, x3_t], dim=1)
        dec_x3 = self.dec_conv3(x_c)
        x2_t = self.tconv2(dec_x3)
        x_c = torch.cat([enc_x2, x2_t], dim=1)
        dec_x2 = self.dec_conv2(x_c)
        x1_t = self.tconv1(dec_x2)
        x_c = torch.cat([enc_x1, x1_t], dim=1)
        dec_x1 = self.dec_conv1(x_c)

        x_final = self.outconv(dec_x1)

        return x_final