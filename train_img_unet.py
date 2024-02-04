import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from matplotlib import pyplot as plt
from pdb import set_trace as st
import cv2
from unet_parts import *
from pytorch_msssim import  ms_ssim, ssim
from metrics import *
import wandb
import sys
import json
import resnet

torch.manual_seed(0)
np.random.seed(0)

N = 700 * 3
ntrain = 1760 #900*2
ntest = 2200-1760#150*2

batch_size = 4
learning_rate = 1e-2#0.001
epochs = 90
iterations = epochs * (ntrain // batch_size)

modes = 20 #12
modes_lat = 12#12
width = 16 #28 #32

r = 10
x_lat = 32#64
x_dep = 955#857
y_lat = 877
y_dep = 844
x_dep_sample = 2

#noise_level = '20dB'
USE_WANDB = True
data_dir1 = '/mlroom/data/lung_set1' #pts/'
data_dir = '/mlroom/data/lung_set1_fix'
EVALUATE = 1-True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_F,  transform=None, mean=None, std=None,  eps=1e-10):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.targets = data[:,1].astype(float)
        self.mean = mean
        self.std = std
        self.eps = eps

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        x = torch.load(self.data[index])
        try:
            x = (x - self.mean) / (self.std + self.eps)
        except:
            print(x.shape)
        return x, self.targets[index]
        
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
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.out_channels)

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
        x = torch.fft.irfft(out_ft, dim=dim, n=out_size) #out_size
        x = self.norm(x)
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
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.out_channels)

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
        x = self.norm(x)

        return x

class MLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP1d, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=mid_channels)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        
    def forward(self, x):
        x = self.mlp1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        x = self.norm2(x)

        return x

class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=mid_channels)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        
    def forward(self, x):
        x = self.mlp1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        x = self.norm2(x)

        return x

def kernel(in_chan=3, up_dim=64):
    """
        Kernel network apply on grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
               # nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=False)
            )
    return layers
    
class FNO(nn.Module):
    def __init__(self, modes1, modes2, width, dep, lat, bilinear=False):
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

        self.p = nn.Linear(4, self.width)  # input channel is 3: (a(x, y), x, y)

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

        self.mlp_areation = torch.nn.Linear(2*self.width, 1)
        self.down1d = nn.MaxPool1d(2)
        self.down3d = nn.MaxPool3d((2,1,1))

        """self.q = MLP1d(self.width, 128, 1)  # output channel is 1: u(x, y)

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
        self.inc = (DoubleConv(self.width*2, self.width))
        #self.down1 = (Down(32, 64))
        #self.down2 = (Down(64, 128))
        #self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        #self.down4 = (Down(256, 512 // factor))
        #self.up1 = (Up(512, 256 // factor, bilinear))
        #self.up2 = (Up(256, 128 // factor, bilinear))
        #self.up3 = (Up(128, 64 // factor, bilinear))
        #self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(self.width, 2))"""
        
        self.mlp_merge_lat0A = torch.nn.Linear(lat, self.width_merge)
        self.mlp_merge_lat1A = torch.nn.Linear(self.width_merge * self.width * 2, self.width)
        #self.norm = nn.LayerNorm([self.width*2, 428, 896])

        self.conv_lat = (DoubleConv(self.width*2, self.width*2))
        self.kernel_dim = width*2 #96
        self.knet = kernel(3, self.kernel_dim)

        #self.init_conv = torch.nn.Conv2d(914, 914, (1,2), stride=(1,2))
        
    def forward(self, x, bmode):
        # x (B, depth, lat, lat, C )
        # bmode (B, )
        #x_img  = x
        '''if np.random.uniform() >=0.8:
            x = x[...,::3,::3]
        elif np.random.uniform() >=0.5:
            x = x[...,::2,::2]
        else:
            pass '''
        ## add noise
        ##
        #x = self.init_conv(x)
        
        batchsize, depth, laterial, laterial1 = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        st()
        x = x.reshape(batchsize, depth, laterial, laterial1, 1)
        grid = self.get_grid3d(x.shape, x.device) # grid (B, depth, lat, lat, 3)
        #xT = x.permute(0,1,3,2,4 ) # x' (B, depth, lat, lat, C )
        x = torch.cat((x, grid), dim=-1) # x (B, depth, lat, lat, 2C+3)
        x = self.p(x) # x (B, depth, lat, lat, width)
        x_input = x.permute(0,4,1,2,3)

        # convolution on depth
        x = x.permute(0,2,3,4,1) # (B, lat, lat, width, depth)
        x = x.reshape(batchsize*laterial*laterial1, -1, depth) # (B*lat*lat, width, depth)

        x1 = self.conv_dep0(x) #FNO 800->200
        x1 = self.mlp_dep0(x1) #MLP
        x2 = self.w_dep0(x) # 1x1 conv : sample/downsample
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv_dep1(x) #, out_size=428)
        x1 = self.mlp_dep1(x1)
        x2 = self.w_dep1(x)
        #x2 = self.down1d(x2)
        #depth = depth//2
        x = x1 + x2
        # (B*lat*lat, width, depth)

        # convolution on laterial
        x = x.reshape(batchsize, laterial, laterial1, -1, depth)
        x = x.permute(0, 4, 3, 1, 2) # (B, depth, width, laterial, laterial)
        x = x.reshape(batchsize*depth, -1, laterial, laterial1) # (B*depth, width, laterial, laterial)

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
        #x_input = self.down3d(x_input)
        x = x.reshape(batchsize, depth, -1, laterial, laterial1)#x = x.permute(1,0,2,3).unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4)
        st()

        x_inter = torch.cat((x, x_input), 1)

        #grid
        grid = self.get_grid3d((x_inter.shape[0], x_inter.shape[2], x_inter.shape[3], x_inter.shape[4]), x_inter.device)
        kx = self.knet(grid)
        # enisum
        kx = kx.reshape(batchsize, -1, x_inter.shape[1]).permute(0, 2, 1)
        x = x_inter.reshape(batchsize, x_inter.shape[1], -1)
        
        x = torch.einsum('bci,bci->bc', kx, x)/(x_inter.shape[-1]*x_inter.shape[-2]*x_inter.shape[-3] )
        x = self.mlp_areation(x)

        return None, x

        #st()
        
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

        ### end of UNET
#         x = F.sigmoid(x)

        return None, x_cls

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


if __name__ == "__main__":
    # data
    x_dep = x_dep // x_dep_sample
    print(x_dep, x_lat)

    '''aeration = []
    seRFraw = []
    for i in range(21):#range(21):
        #seRFraw1 = torch.load(data_dir+"seRFraw_"+str(i)+".pt")#torch.load('/raid/peter/us_lung/focused/seRFraw_'+str(noise_level)+'.pt') #torch.load('/raid/peter/us_lung/seRFtd.pt') #/raid/zongyi/ultrasound/seRFraw.pt')
        seRFraw1 = torch.load(data_dir+'/addRF_'+str(i)+".pt")
        seRFraw1 = seRFraw1[:,::x_dep_sample,:,::2]#[:,:x_dep]
        #seRFraw = seRFraw[:,:, ::3,::3]
        #rhomap = torch.load('/raid/peter/us_lung/focused/rhomap.pt')
        aeration1 = torch.load(data_dir+"/aeration_"+str(i)+".pt")#torch.load('/raid/peter/us_lung/focused/aeration.pt')
        
        aeration.append(aeration1)
        seRFraw.append(seRFraw1)

    for i in range(24):#range(21):
        seRFraw1 = torch.load(data_dir1+'/addRF_'+str(i)+".pt")
        seRFraw1 = seRFraw1[:,::x_dep_sample,:,::2]#[:,:x_dep]
        #aeration1 = torch.load(data_dir1+"aeration_"+str(i)+".pt")#torch.load('/raid/peter/us_lung/focused/aeration.pt')
        
        seRFraw.append(seRFraw1)
        
    aeration1 = torch.load(data_dir1+'/aerations_.pt') #torch.cat(aeration, dim=0)
    aeration.append(aeration1)

    seRFraw = torch.cat(seRFraw, dim=0)
    aeration = torch.cat(aeration, dim=0)'''


    # process rhomap
    '''rhomap = rhomap.permute(0,2,1)
    rhomap = rhomap[:,469:811, 62:814]
    rhomap = (rhomap!=0).long()'''
    #rhomap = nn.functional.relu(rhomap-850)
        
    #print(seRFraw.shape, aeration.shape)

    #index = torch.load("index2100.pt")#torch.randperm(N)
    '''try:
        train_index = index[:ntrain]
        test_index = index[ntrain:]
        seRFraw_train = seRFraw[train_index]
        aeration_train = aeration[train_index]
    except:
        print('fail!')
        train_index = np.arange(100)
        test_index = train_index
        seRFraw_train = seRFraw[train_index]
        aeration_train = aeration[train_index]'''
    mean_file = torch.load('raw_mean.pt')
    std_file = torch.load('raw_std.pt')
    
    train_set = Dataset('add_a0_4500_train.json', mean = mean_file, std=std_file)
    test_set = Dataset('add_a0_4500_test.json', mean = mean_file, std=std_file)
    
    '''seRFraw_test = seRFraw[test_index]
    #rhomap_test = rhomap[test_index]
    aeration_test = aeration[test_index]
    
    x_normalizer = UnitGaussianNormalizer(seRFraw_train)
    seRFraw_train = x_normalizer.encode(seRFraw_train)
    seRFraw_test = x_normalizer.encode(seRFraw_test)'''
    
    #y_normalizer = UnitGaussianNormalizer(rhomap_train)
    #rhomap_train = y_normalizer.encode(rhomap_train)
    #rhomap_test = y_normalizer.encode(rhomap_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True)#False)
    
    '''train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(seRFraw_train, aeration_train), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(seRFraw_test, aeration_test), batch_size=1,
                                              shuffle=True)#False)'''
    ntrain = len(train_loader)
    ntest = len(test_loader)

    # model
    model = resnet.resnet18(num_classes=1).cuda()#FNO(modes, modes_lat, width, dep=x_dep, lat=x_lat).cuda()
    #model.load_state_dict(torch.load("f32_norm_0.1_SNmultask_N900_m12_w32_e100.pt"))

    print(count_params(model))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    myloss = LpLoss(size_average=False)
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    
    
    
    criterion = nn.CrossEntropyLoss()
    cls_loss = 0

    if USE_WANDB:
        wandb.init(
            dir="./wandb_files",
            entity="peterwg",
            project='fno_us_noise',
            name=str(sys.argv[1]),
            #id="{}_{}_{}".format('fno_us_noise', params.exp_name, random_date),
        )

        
    for ep in range(epochs):
        t1 = default_timer()

        if not EVALUATE:
            model.train()
            train_l2 = 0
            for raw, aera in train_loader:
                raw, aera = raw.cuda(), aera.cuda()
    
                optimizer.zero_grad()
                aera_out = model(raw)
                
                #out, aera_out = model(raw, bmode=None)
                cls_loss = l1loss(aera_out.view(batch_size, -1).float(), aera.view(batch_size, -1).float()) + l2loss(aera_out.view(batch_size, -1).float(), aera.view(batch_size, -1).float())
                #st()
                #cls_loss = myloss(aera_out.view(batch_size, -1), aera.view(batch_size, -1))
                #loss = criterion(out, rho)
                #loss = l1loss(out.squeeze(1),rho)
                #loss += 1- ms_ssim( out, rho.unsqueeze(1), data_range=1, size_average=True )
                loss = cls_loss
                loss.backward()
    
                optimizer.step()
                scheduler.step()
                train_l2 += loss.item()
        if EVALUATE:
            pass
            #model.load_state_dict(torch.load('fw_set1.pt'))
        model.eval()
        test_l2 = 0.0
        nsme_a = 0.0
        ssim_a = 0.0
        ms_ssim_a = 0.0
        psnr_a = 0.0

        with torch.no_grad():
            pred = np.zeros((ntest,))
            truth = np.zeros((ntest,))
            for i, (raw, aera) in enumerate(test_loader):
                raw, aera = raw.cuda(), aera.cuda()
    
                aera_out = model(raw)
    
                #cur_test_loss = ms_ssim( out.argmax(1).unsqueeze(1).float(), rho.unsqueeze(1).float(), data_range=1, size_average=True ) #criterion(out, rho)
                '''cur_test_loss = 1*l1loss(out.squeeze(1),rho)
                #cur_test_loss += 0*l2loss(out.squeeze(1),rho)
                cur_test_loss +=  1- ms_ssim( out, rho.unsqueeze(1), data_range=1, size_average=True )'''
                test_l2 += ( l1loss(aera_out.view(batch_size, -1), aera.view(batch_size, -1)) + l2loss(aera_out.view(batch_size, -1), aera.view(batch_size, -1)) ).item() #myloss(aera_out.view(1, -1), aera.view(1, -1)).item()
                #nsme_a += nsme(rho[0].float(), out[0].argmax(0)).item()
                #ssim_a += ssim( out.argmax(1).unsqueeze(1).float(), rho.unsqueeze(1).float(), data_range=1, size_average=True ).item()
                #ms_ssim_a += ms_ssim( out.argmax(1).unsqueeze(1).float(), rho.unsqueeze(1).float(), data_range=1, size_average=True ).item()
                #psnr_a += psnr(rho[0].float(), out[0].argmax(0)).item()
                try:
                    pred[i*batch_size:(i+1)*batch_size] = aera_out.cpu().squeeze().numpy()#.view(1, -1).item()
                    truth[i*batch_size:(i+1)*batch_size] = aera.cpu().squeeze().numpy()#.view(1, -1).item()
                except:
                    pass
        error = np.abs(pred - truth)
        #cls_loss = 0 #cls_loss.item()/ntrain
        if not EVALUATE:
            train_l2 /= ntrain
        else:
            train_l2 = 0
        test_l2 /= ntest

        
        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2)
        if USE_WANDB:
            wandb.log({"train_l2": train_l2})
            wandb.log({"test_l2": test_l2})
            wandb.log({"error_mean": np.mean(error)})
            wandb.log({"error_max": np.max(error)})
            wandb.log({"error_0.1": (error<0.1).sum()/len(test_loader)})
            wandb.log({"error_1": (error<1).sum()/len(test_loader)})
            wandb.log({"error_10": (error<10).sum()/len(test_loader)})
        torch.save(model.state_dict(), 'current_.pt') 
        #cv2.imwrite("results/b3cls_norm_0.1_SNunet_mul_"+str(ep)+'.png',torch.cat((rho[0].squeeze().cpu(), out[0].argmax(0).squeeze().cpu()),0).int().numpy()*255)
        #torch.save(model.state_dict(), 'current_.pt') 
        #cv2.imwrite("results/Nunet_mul_"+str(ep)+'.png',torch.cat((y_normalizer.decode(rho.squeeze().cpu()), y_normalizer.decode(out.squeeze().cpu())),0).numpy())
    torch.save(model.state_dict(), 'fw_set1.pt') 
