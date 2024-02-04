"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from unet_parts_bn import *

torch.manual_seed(0)
np.random.seed(0)

width = 32

################################################################
# fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim=-1, out_size=None):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=dim)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        if out_size == None:
            out_size = x.size(-1)
        x = torch.fft.irfft(out_ft, dim=dim, n=out_size)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim=(-2,-1), out_size=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, dim=dim)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if out_size == None:
            out_size = (x.size(-2), x.size(-1))
        x = torch.fft.irfft2(out_ft, dim=dim, s=out_size)
        return x

class MLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP1d, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO(nn.Module):
    def __init__(self, modes1, modes2, width, dep, lat):
        super(FNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes_dep = modes1
        self.modes_lat = modes2
        self.width = width
        self.width_merge = 8

        self.p = nn.Linear(5, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_dep0 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.conv_dep1 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.mlp_dep0 = MLP1d(self.width, self.width, self.width)
        self.mlp_dep1 = MLP1d(self.width, self.width, self.width)
        self.w_dep0 = nn.Conv1d(self.width, self.width, 1)
        self.w_dep1 = nn.Conv1d(self.width, self.width, 1)

        self.conv_lat0 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.conv_lat1 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.mlp_lat0 = MLP2d(self.width, self.width, self.width)
        self.mlp_lat1 = MLP2d(self.width, self.width, self.width)
        self.w_lat0 = nn.Conv2d(self.width, self.width, 1)
        self.w_lat1 = nn.Conv2d(self.width, self.width, 1)

        self.mlp_merge_lat0 = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_lat1 = torch.nn.Linear(self.width_merge * self.width, self.width)
        self.mlp_merge_latp0 = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_latp1 = torch.nn.Linear(self.width_merge * self.width, self.width)
        self.mlp_merge_dep0 = torch.nn.Linear(dep, self.width_merge)
        self.mlp_merge_dep1 = torch.nn.Linear(self.width_merge * self.width, self.width)

        self.mlp_areation = torch.nn.Linear(self.width, 1)

        self.q = MLP1d(self.width, 128, 1)  # output channel is 1: u(x, y)

    def forward(self, x, bmode):
        # x (B, depth, lat, lat, C )
        # bmode (B, )

        batchsize, depth, laterial, = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(batchsize, depth, laterial, laterial, 1)
        grid = self.get_grid3d(x.shape, x.device) # grid (B, depth, lat, lat, 3)
        xT = x.permute(0,1,3,2,4 ) # x' (B, depth, lat, lat, C )
        x = torch.cat((x, xT, grid), dim=-1) # x (B, depth, lat, lat, 2C+3)
        x = self.p(x) # x (B, depth, lat, lat, width)
        x_input = x.permute(0,4,1,2,3)

        # convolution on depth
        x = x.permute(0,2,3,4,1) # (B, lat, lat, width, depth)
        x = x.reshape(batchsize*laterial*laterial, -1, depth) # (B*lat*lat, width, depth)

        x1 = self.conv_dep0(x)
        x1 = self.mlp_dep0(x1)
        x2 = self.w_dep0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_dep1(x)
        x1 = self.mlp_dep1(x1)
        x2 = self.w_dep1(x)
        x = x1 + x2
        # (B*lat*lat, width, depth)

        # convolution on laterial
        x = x.reshape(batchsize, laterial, laterial, -1, depth)
        x = x.permute(0, 4, 3, 1, 2) # (B, depth, width, laterial, laterial)
        x = x.reshape(batchsize*depth, -1, laterial, laterial) # (B*depth, width, laterial, laterial)

        x1 = self.conv_lat0(x)
        x1 = self.mlp_lat0(x1)
        x2 = self.w_lat0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_lat1(x)
        x1 = self.mlp_lat1(x1)
        x2 = self.w_lat1(x)
        x = x1 + x2
        # (B*depth, width, laterial, laterial)


        # # B-mode  (B, depth, lat, C)
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # x = self.p(x)
        # x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])
        #
        # x1 = self.conv0(x)
        # x1 = self.mlp0(x1)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)
        #
        # x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # merge the laterial dim
#         x = x.reshape(batchsize, depth, -1, laterial, laterial)  # (B, depth, width, laterial, laterial)
#         x = x.permute(0, 2, 1, 3, 4) # (B, width, depth, laterial, laterial)
#         x = x + x_input
        x = x_input

        x = self.mlp_merge_lat0(x) # (B, width, depth, laterial, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(batchsize, depth, laterial, -1) # (B, depth, laterial, width*10)
        x = self.mlp_merge_lat1(x) # (B, depth, laterial, width)
        x = x.permute(0, 3, 1, 2) # (B, width, depth, laterial)

        x = self.mlp_merge_latp0(x) # (B, width, depth, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, depth, -1) # (B, depth, width*10)
        x = self.mlp_merge_latp1(x) # (B, depth, laterial, width)
        x = x.permute(0, 2, 1) # (B, width, depth)

        x = self.mlp_merge_dep0(x) # (B, width, 10)
        x = F.gelu(x)
        x = x.reshape(batchsize, -1) # (B, width*10)
        x = self.mlp_merge_dep1(x) # (B, width*10)
        x = F.gelu(x)
        x = self.mlp_areation(x)
#         x = F.sigmoid(x)

        return x

    def get_grid2d(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def get_grid3d(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

class FNO_gen2(nn.Module):
    def __init__(self, modes1, modes2, width, dep, lat):
        super(FNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes_dep = modes1
        self.modes_lat = modes2
        self.width = width
        self.width_merge = 8

        self.p = nn.Linear(5, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_dep0 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.conv_dep1 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.mlp_dep0 = MLP1d(self.width, self.width, self.width)
        self.mlp_dep1 = MLP1d(self.width, self.width, self.width)
        self.w_dep0 = nn.Conv1d(self.width, self.width, 1)
        self.w_dep1 = nn.Conv1d(self.width, self.width, 1)

        self.conv_lat0 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.conv_lat1 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.mlp_lat0 = MLP2d(self.width, self.width, self.width)
        self.mlp_lat1 = MLP2d(self.width, self.width, self.width)
        self.w_lat0 = nn.Conv2d(self.width, self.width, 1)
        self.w_lat1 = nn.Conv2d(self.width, self.width, 1)

        self.mlp_merge_lat0 = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_lat1 = torch.nn.Linear(self.width_merge * self.width * 2, self.width)
        self.mlp_merge_latp0 = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_latp1 = torch.nn.Linear(self.width_merge * self.width, self.width)
        self.mlp_merge_dep0 = torch.nn.Linear(dep, self.width_merge)
        self.mlp_merge_dep1 = torch.nn.Linear(self.width_merge * self.width, self.width)

        self.mlp_areation = torch.nn.Linear(self.width, 1)

        self.q = MLP1d(self.width, 128, 1)  # output channel is 1: u(x, y)

    def forward(self, x): #, bmode):
        # x (B, depth, lat, lat, C )
        # bmode (B, )

        batchsize, depth, laterial, = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(batchsize, depth, laterial, laterial, 1)
        grid = self.get_grid3d(x.shape, x.device) # grid (B, depth, lat, lat, 3)
        xT = x.permute(0,1,3,2,4 ) # x' (B, depth, lat, lat, C )
        x = torch.cat((x, xT, grid), dim=-1) # x (B, depth, lat, lat, 2C+3)
        x = self.p(x) # x (B, depth, lat, lat, width)
        x_input = x.permute(0,4,1,2,3) 

        # convolution on depth
        x = x.permute(0,2,3,4,1) # (B, lat, lat, width, depth)
        x = x.reshape(batchsize*laterial*laterial, -1, depth) # (B*lat*lat, width, depth)

        x1 = self.conv_dep0(x)
        x1 = self.mlp_dep0(x1)
        x2 = self.w_dep0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_dep1(x)
        x1 = self.mlp_dep1(x1)
        x2 = self.w_dep1(x)
        x = x1 + x2
        # (B*lat*lat, width, depth)

        # convolution on laterial
        x = x.reshape(batchsize, laterial, laterial, -1, depth)
        x = x.permute(0, 4, 3, 1, 2) # (B, depth, width, laterial, laterial)
        x = x.reshape(batchsize*depth, -1, laterial, laterial) # (B*depth, width, laterial, laterial)

        x1 = self.conv_lat0(x)
        x1 = self.mlp_lat0(x1)
        x2 = self.w_lat0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_lat1(x)
        x1 = self.mlp_lat1(x1)
        x2 = self.w_lat1(x)
        x = x1 + x2
        # (B*depth, width, laterial, laterial)


        # # B-mode  (B, depth, lat, C)
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # x = self.p(x)
        # x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])
        #
        # x1 = self.conv0(x)
        # x1 = self.mlp0(x1)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)
        #
        # x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # merge the laterial dim
#         x = x.reshape(batchsize, depth, -1, laterial, laterial)  # (B, depth, width, laterial, laterial)
#         x = x.permute(0, 2, 1, 3, 4) # (B, width, depth, laterial, laterial)
#         x = x + x_input
        #x = x_input # question mark?
        x = x.permute(1,0,2,3).unsqueeze(0)
        x = torch.cat((x, x_input), 1)
        #x = torch.cat((x.permute(1,0,2,3).unsqueeze(0), x_input), 1)
        
        x = self.mlp_merge_lat0(x) # (B, width, depth, laterial, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(batchsize, depth, laterial, -1) # (B, depth, laterial, width*10)
        x = self.mlp_merge_lat1(x) # (B, depth, laterial, width)
        x = x.permute(0, 3, 1, 2) # (B, width, depth, laterial)

        x = self.mlp_merge_latp0(x) # (B, width, depth, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, depth, -1) # (B, depth, width*10)
        x = self.mlp_merge_latp1(x) # (B, depth, laterial, width)
        x = x.permute(0, 2, 1) # (B, width, depth)

        x = self.mlp_merge_dep0(x) # (B, width, 10)
        x = F.gelu(x)
        x = x.reshape(batchsize, -1) # (B, width*10)
        x = self.mlp_merge_dep1(x) # (B, width*10)
        x = F.gelu(x)
        x = self.mlp_areation(x)
#         x = F.sigmoid(x)

        return x
        
    def get_grid2d(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def get_grid3d(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

class FNO2(nn.Module):
    def __init__(self, modes1, modes2, width, dep, lat):
        super(FNO2, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes_dep = modes1
        self.modes_lat = modes2
        self.width = width
        self.width_merge = 8

        self.p = nn.Linear(5, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_dep0 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.conv_dep1 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.mlp_dep0 = MLP1d(self.width, self.width, self.width)
        self.mlp_dep1 = MLP1d(self.width, self.width, self.width)
        self.w_dep0 = nn.Conv1d(self.width, self.width, 1)
        self.w_dep1 = nn.Conv1d(self.width, self.width, 1)

        self.conv_lat0 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.conv_lat1 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.mlp_lat0 = MLP2d(self.width, self.width, self.width)
        self.mlp_lat1 = MLP2d(self.width, self.width, self.width)
        self.w_lat0 = nn.Conv2d(self.width, self.width, 1)
        self.w_lat1 = nn.Conv2d(self.width, self.width, 1)

        self.mlp_merge_lat0 = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_lat1 = torch.nn.Linear(self.width_merge * self.width * 2, self.width)
        self.mlp_merge_latp0 = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_latp1 = torch.nn.Linear(self.width_merge * self.width, self.width)
        self.mlp_merge_dep0 = torch.nn.Linear(dep, self.width_merge)
        self.mlp_merge_dep1 = torch.nn.Linear(self.width_merge * self.width, self.width)

        self.mlp_areation = torch.nn.Linear(self.width, 1)

        self.q = MLP1d(self.width, 128, 1)  # output channel is 1: u(x, y)

    def forward(self, x): #, bmode):
        # x (B, depth, lat, lat, C )
        # bmode (B, )

        batchsize, depth, laterial, = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(batchsize, depth, laterial, laterial, 1)
        grid = self.get_grid3d(x.shape, x.device) # grid (B, depth, lat, lat, 3)
        xT = x.permute(0,1,3,2,4 ) # x' (B, depth, lat, lat, C )
        x = torch.cat((x, xT, grid), dim=-1) # x (B, depth, lat, lat, 2C+3)
        x = self.p(x) # x (B, depth, lat, lat, width)
        x_input = x.permute(0,4,1,2,3) 

        # convolution on depth
        x = x.permute(0,2,3,4,1) # (B, lat, lat, width, depth)
        x = x.reshape(batchsize*laterial*laterial, -1, depth) # (B*lat*lat, width, depth)

        x1 = self.conv_dep0(x)
        x1 = self.mlp_dep0(x1)
        x2 = self.w_dep0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_dep1(x)
        x1 = self.mlp_dep1(x1)
        x2 = self.w_dep1(x)
        x = x1 + x2
        # (B*lat*lat, width, depth)

        # convolution on laterial
        x = x.reshape(batchsize, laterial, laterial, -1, depth)
        x = x.permute(0, 4, 3, 1, 2) # (B, depth, width, laterial, laterial)
        x = x.reshape(batchsize*depth, -1, laterial, laterial) # (B*depth, width, laterial, laterial)

        x1 = self.conv_lat0(x)
        x1 = self.mlp_lat0(x1)
        x2 = self.w_lat0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_lat1(x)
        x1 = self.mlp_lat1(x1)
        x2 = self.w_lat1(x)
        x = x1 + x2
        # (B*depth, width, laterial, laterial)


        # # B-mode  (B, depth, lat, C)
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # x = self.p(x)
        # x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])
        #
        # x1 = self.conv0(x)
        # x1 = self.mlp0(x1)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)
        #
        # x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # merge the laterial dim
#         x = x.reshape(batchsize, depth, -1, laterial, laterial)  # (B, depth, width, laterial, laterial)
#         x = x.permute(0, 2, 1, 3, 4) # (B, width, depth, laterial, laterial)
#         x = x + x_input
        #x = x_input # question mark?
        x = x.permute(1,0,2,3).unsqueeze(0)
        x = torch.cat((x, x_input), 1)
        #x = torch.cat((x.permute(1,0,2,3).unsqueeze(0), x_input), 1)
        
        x = self.mlp_merge_lat0(x) # (B, width, depth, laterial, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(batchsize, depth, laterial, -1) # (B, depth, laterial, width*10)
        x = self.mlp_merge_lat1(x) # (B, depth, laterial, width)
        x = x.permute(0, 3, 1, 2) # (B, width, depth, laterial)

        x = self.mlp_merge_latp0(x) # (B, width, depth, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, depth, -1) # (B, depth, width*10)
        x = self.mlp_merge_latp1(x) # (B, depth, laterial, width)
        x = x.permute(0, 2, 1) # (B, width, depth)

        x = self.mlp_merge_dep0(x) # (B, width, 10)
        x = F.gelu(x)
        x = x.reshape(batchsize, -1) # (B, width*10)
        x = self.mlp_merge_dep1(x) # (B, width*10)
        x = F.gelu(x)
        x = self.mlp_areation(x)
#         x = F.sigmoid(x)

        return x
        
    def get_grid2d(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def get_grid3d(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

class FNOim(nn.Module):
    def __init__(self, modes1, modes2, width, dep, lat, bilinear=False):
        super(FNOim, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes_dep = modes1
        self.modes_lat = modes2
        self.width = width
        self.width_merge = 8

        self.p = nn.Linear(5, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_dep0 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.conv_dep1 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.mlp_dep0 = MLP1d(self.width, self.width, self.width)
        self.mlp_dep1 = MLP1d(self.width, self.width, self.width)
        self.w_dep0 = nn.Conv1d(self.width, self.width, 1)
        self.w_dep1 = nn.Conv1d(self.width, self.width, 1)

        self.conv_lat0 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.conv_lat1 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.mlp_lat0 = MLP2d(self.width, self.width, self.width)
        self.mlp_lat1 = MLP2d(self.width, self.width, self.width)
        self.w_lat0 = nn.Conv2d(self.width, self.width, 1)
        self.w_lat1 = nn.Conv2d(self.width, self.width, 1)

        self.mlp_merge_lat0 = torch.nn.Linear(lat* lat, 896)#(lat, self.width_merge)
        self.mlp_merge_lat1 = torch.nn.Linear(896, 896)#(self.width_merge * self.width *2, self.width)
        self.mlp_merge_latp0 = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_latp1 = torch.nn.Linear(self.width_merge * self.width, self.width)
        self.mlp_merge_dep0 = torch.nn.Linear(dep//2, self.width_merge) #!
        self.mlp_merge_dep1 = torch.nn.Linear(self.width_merge * self.width, self.width)

        self.mlp_areation = torch.nn.Linear(self.width, 1)

        self.q = MLP1d(self.width, 128, 1)  # output channel is 1: u(x, y)

        self.down1d = nn.MaxPool1d(2)
        self.down3d = nn.MaxPool3d((2,1,1))
        self.upsample = nn.Upsample(scale_factor=(1, 14))

        '''self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))

        self.up1 = (Up(256, 256 , bilinear))
        self.up2 = (Up(256, 128, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        
        self.outc = (OutConv(256, 1))'''
        self.inc = (DoubleConv(64, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, 1))

    def forward(self, x, bmode):
        # x (B, depth, lat, lat, C )
        # bmode (B, )
        batchsize, depth, laterial, = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(batchsize, depth, laterial, laterial, 1)
        grid = self.get_grid3d(x.shape, x.device) # grid (B, depth, lat, lat, 3)
        xT = x.permute(0,1,3,2,4 ) # x' (B, depth, lat, lat, C )
        x = torch.cat((x, xT, grid), dim=-1) # x (B, depth, lat, lat, 2C+3)
        x = self.p(x) # x (B, depth, lat, lat, width)
        x_input = x.permute(0,4,1,2,3)

        # convolution on depth
        x = x.permute(0,2,3,4,1) # (B, lat, lat, width, depth)
        x = x.reshape(batchsize*laterial*laterial, -1, depth) # (B*lat*lat, width, depth)

        x1 = self.conv_dep0(x) #FNO 800->200
        x1 = self.mlp_dep0(x1) #MLP
        x2 = self.w_dep0(x) # 1x1 conv : sample/downsample
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_dep1(x, out_size=428)
        x1 = self.mlp_dep1(x1)
        x2 = self.w_dep1(x)
        x2 = self.down1d(x2)
        depth = depth//2
        x = x1 + x2
        # (B*lat*lat, width, depth)

        # convolution on laterial
        x = x.reshape(batchsize, laterial, laterial, -1, depth)
        x = x.permute(0, 4, 3, 1, 2) # (B, depth, width, laterial, laterial)
        x = x.reshape(batchsize*depth, -1, laterial, laterial) # (B*depth, width, laterial, laterial)

        x1 = self.conv_lat0(x)
        x1 = self.mlp_lat0(x1)
        x2 = self.w_lat0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_lat1(x)
        x1 = self.mlp_lat1(x1)
        x2 = self.w_lat1(x)
        x = x1 + x2
        # (B*depth, width, laterial, laterial)


        # # B-mode  (B, depth, lat, C)
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # x = self.p(x)
        # x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])
        #
        # x1 = self.conv0(x)
        # x1 = self.mlp0(x1)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)
        #
        # x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # merge the laterial dim
#         x = x.reshape(batchsize, depth, -1, laterial, laterial)  # (B, depth, width, laterial, laterial)
#         x = x.permute(0, 2, 1, 3, 4) # (B, width, depth, laterial, laterial)
#         x = x + x_input
        x_input = self.down3d(x_input)
        x = x.permute(1,0,2,3).unsqueeze(0)
        x = torch.cat((x, x_input), 1)

        x = x.reshape(batchsize,  self.width*2, depth, -1)
        x = self.mlp_merge_lat0(x) # (B, width, depth, laterial, 10)
        x = F.gelu(x)
        x = self.mlp_merge_lat1(x)
        
        '''x = self.mlp_merge_lat0(x) # (B, width, depth, laterial, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(batchsize, depth, laterial, -1) # (B, depth, laterial, width*10)
        x = self.mlp_merge_lat1(x) # (B, depth, laterial, width)
        x = x.permute(0, 3, 1, 2) # (B, width, depth, laterial)'''
        
        """
        # for aeration cls. skip for now
        x = self.mlp_merge_latp0(x) # (B, width, depth, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, depth, -1) # (B, depth, width*10)
        x = self.mlp_merge_latp1(x) # (B, depth, laterial, width)
        x = x.permute(0, 2, 1) # (B, width, depth)

        x = self.mlp_merge_dep0(x) # (B, width, 10)
        x = F.gelu(x)
        x = x.reshape(batchsize, -1) # (B, width*10)
        x = self.mlp_merge_dep1(x) # (B, width*10)
        x = F.gelu(x)
        x = self.mlp_areation(x)"""
        
        #x1 = self.upsample(x)

        ### UNET
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x_map = self.outc(x)
        #crop
        x_map = x_map[..., 43:385 , 72:824]
        ### end of UNET
#         x = F.sigmoid(x)

        return x_map

    def get_grid2d(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def get_grid3d(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
    
class FNO_seg(nn.Module):
    def __init__(self, modes1, modes2, width, dep, lat, bilinear=False):
        super(FNO_seg, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes_dep = modes1
        self.modes_lat = modes2
        self.width = width
        self.width_merge = 8

        self.p = nn.Linear(5, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_dep0 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.conv_dep1 = SpectralConv1d(self.width, self.width, self.modes_dep)
        self.mlp_dep0 = MLP1d(self.width, self.width, self.width)
        self.mlp_dep1 = MLP1d(self.width, self.width, self.width)
        self.w_dep0 = nn.Conv1d(self.width, self.width, 1)
        self.w_dep1 = nn.Conv1d(self.width, self.width, 1)

        self.conv_lat0 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.conv_lat1 = SpectralConv2d(self.width, self.width, self.modes_lat, self.modes_lat)
        self.mlp_lat0 = MLP2d(self.width, self.width, self.width)
        self.mlp_lat1 = MLP2d(self.width, self.width, self.width)
        self.w_lat0 = nn.Conv2d(self.width, self.width, 1)
        self.w_lat1 = nn.Conv2d(self.width, self.width, 1)

        self.mlp_merge_lat0 = torch.nn.Linear(lat* lat, 896)#(lat, self.width_merge)
        self.mlp_merge_lat1 = torch.nn.Linear(896, 896)#(self.width_merge * self.width *2, self.width)
        self.mlp_merge_latp0 = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_latp1 = torch.nn.Linear(self.width_merge * self.width, self.width)
        self.mlp_merge_dep0 = torch.nn.Linear(dep//2, self.width_merge) #!
        self.mlp_merge_dep1 = torch.nn.Linear(self.width_merge * self.width, self.width)

        self.mlp_areation = torch.nn.Linear(self.width, 1)

        self.q = MLP1d(self.width, 128, 1)  # output channel is 1: u(x, y)

        self.down1d = nn.MaxPool1d(2)
        self.down3d = nn.MaxPool3d((2,1,1))
        self.upsample = nn.Upsample(scale_factor=(1, 14))

        '''self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))

        self.up1 = (Up(256, 256 , bilinear))
        self.up2 = (Up(256, 128, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        
        self.outc = (OutConv(256, 1))'''
        self.inc = (DoubleConv(64, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, 2))
        
        self.mlp_merge_lat0A = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_lat1A = torch.nn.Linear(self.width_merge * self.width * 2, self.width)
        self.norm = nn.LayerNorm([64, 428, 896])
        
    def forward(self, x, bmode):
        # x (B, depth, lat, lat, C )
        # bmode (B, )
        batchsize, depth, laterial, = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(batchsize, depth, laterial, laterial, 1)
        grid = self.get_grid3d(x.shape, x.device) # grid (B, depth, lat, lat, 3)
        xT = x.permute(0,1,3,2,4 ) # x' (B, depth, lat, lat, C )
        x = torch.cat((x, xT, grid), dim=-1) # x (B, depth, lat, lat, 2C+3)
        x = self.p(x) # x (B, depth, lat, lat, width)
        x_input = x.permute(0,4,1,2,3)

        # convolution on depth
        x = x.permute(0,2,3,4,1) # (B, lat, lat, width, depth)
        x = x.reshape(batchsize*laterial*laterial, -1, depth) # (B*lat*lat, width, depth)

        x1 = self.conv_dep0(x) #FNO 800->200
        x1 = self.mlp_dep0(x1) #MLP
        x2 = self.w_dep0(x) # 1x1 conv : sample/downsample
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_dep1(x, out_size=428)
        x1 = self.mlp_dep1(x1)
        x2 = self.w_dep1(x)
        x2 = self.down1d(x2)
        depth = depth//2
        x = x1 + x2
        # (B*lat*lat, width, depth)

        # convolution on laterial
        x = x.reshape(batchsize, laterial, laterial, -1, depth)
        x = x.permute(0, 4, 3, 1, 2) # (B, depth, width, laterial, laterial)
        x = x.reshape(batchsize*depth, -1, laterial, laterial) # (B*depth, width, laterial, laterial)

        x1 = self.conv_lat0(x)
        x1 = self.mlp_lat0(x1)
        x2 = self.w_lat0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_lat1(x)
        x1 = self.mlp_lat1(x1)
        x2 = self.w_lat1(x)
        x = x1 + x2
        # (B*depth, width, laterial, laterial)


        # # B-mode  (B, depth, lat, C)
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # x = self.p(x)
        # x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])
        #
        # x1 = self.conv0(x)
        # x1 = self.mlp0(x1)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)
        #
        # x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # merge the laterial dim
#         x = x.reshape(batchsize, depth, -1, laterial, laterial)  # (B, depth, width, laterial, laterial)
#         x = x.permute(0, 2, 1, 3, 4) # (B, width, depth, laterial, laterial)
#         x = x + x_input
        x_input = self.down3d(x_input)
        x = x.reshape(batchsize, depth, -1, laterial, laterial)#x = x.permute(1,0,2,3).unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4)
        x_inter = torch.cat((x, x_input), 1)

        #clss
        x = self.mlp_merge_lat0A(x_inter) # (B, width, depth, laterial, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(batchsize, depth, laterial, -1) # (B, depth, laterial, width*10)
        x = self.mlp_merge_lat1A(x) # (B, depth, laterial, width)
        x = x.permute(0, 3, 1, 2) # (B, width, depth, laterial)

        x = self.mlp_merge_latp0(x) # (B, width, depth, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, depth, -1) # (B, depth, width*10)
        x = self.mlp_merge_latp1(x) # (B, depth, laterial, width)
        x = x.permute(0, 2, 1) # (B, width, depth)

        x = self.mlp_merge_dep0(x) # (B, width, 10)
        x = F.gelu(x)
        x = x.reshape(batchsize, -1) # (B, width*10)
        x = self.mlp_merge_dep1(x) # (B, width*10)
        x = F.gelu(x)
        x_cls = self.mlp_areation(x)
        #clss

        x = x_inter.reshape(batchsize, width*2, depth, -1)
        x = self.mlp_merge_lat0(x) # (B, width, depth, laterial, 10)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.mlp_merge_lat1(x)
        
        '''x = self.mlp_merge_lat0(x) # (B, width, depth, laterial, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(batchsize, depth, laterial, -1) # (B, depth, laterial, width*10)
        x = self.mlp_merge_lat1(x) # (B, depth, laterial, width)
        x = x.permute(0, 3, 1, 2) # (B, width, depth, laterial)'''
        
        """
        # for aeration cls. skip for now
        x = self.mlp_merge_latp0(x) # (B, width, depth, 10)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, depth, -1) # (B, depth, width*10)
        x = self.mlp_merge_latp1(x) # (B, depth, laterial, width)
        x = x.permute(0, 2, 1) # (B, width, depth)

        x = self.mlp_merge_dep0(x) # (B, width, 10)
        x = F.gelu(x)
        x = x.reshape(batchsize, -1) # (B, width*10)
        x = self.mlp_merge_dep1(x) # (B, width*10)
        x = F.gelu(x)
        x = self.mlp_areation(x)"""
        
        #x1 = self.upsample(x)

        ### UNET
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x_map = self.outc(x)
        #crop
        x_map = x_map[..., 43:385 , 72:824]
        ### end of UNET
#         x = F.sigmoid(x)

        return x_map, x_cls

    def get_grid2d(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def get_grid3d(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)