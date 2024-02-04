

import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

################################################################
# configs
################################################################
Data_PATH20 = '/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a20_mnr1_35_nn11_20.mat'
Data_PATH48 = '/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a48_mnr1_35_nn11_20.mat'
Data_PATH69 = '/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a69_mnr1_35_nn11_20.mat'

N = 350 * 3
ntrain = 900
ntest = 150

################################################################
# load data and data normalization
################################################################
# data = scipy.io.loadmat(Data_PATH20)


data = h5py.File('/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a20_raw.mat', 'r')
seRFraw = data['a20_raw'] #(N, depth, x_lat, x_lat')
seRFraw1 = torch.tensor(seRFraw, dtype=torch.float).permute(3,2,1,0)
print('finish loading seRFraw', seRFraw.shape)

data = h5py.File('/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a20_mnr1_35_nn11_20.mat', 'r')
rhomap = data['GT']['rhomap'] #(N, y_lat, depth)
rhomap1 = torch.tensor(rhomap, dtype=torch.float).permute(2,1,0)
print('finish loading rhomap', rhomap.shape)

aeration = data['GT']['aeration'] #(N,)
aeration1 = torch.tensor(aeration, dtype=torch.float).reshape(-1,)
print('finish loading aeration', aeration.shape)



data = h5py.File('/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a48_raw.mat', 'r')
seRFraw = data['a48_raw'] #(N, depth, x_lat, x_lat')
seRFraw2 = torch.tensor(seRFraw, dtype=torch.float).permute(3,2,1,0)
print('finish loading seRFraw', seRFraw.shape)

data = h5py.File('/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a48_mnr1_35_nn11_20.mat', 'r')
rhomap = data['GT']['rhomap'] #(N, y_lat, depth)
rhomap2 = torch.tensor(rhomap, dtype=torch.float).permute(2,1,0)
print('finish loading rhomap', rhomap.shape)

aeration = data['GT']['aeration'] #(N,)
aeration2 = torch.tensor(aeration, dtype=torch.float).reshape(-1,)
print('finish loading aeration', aeration.shape)




data = h5py.File('/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a69_raw.mat', 'r')
seRFraw = data['a69_raw'] #(N, depth, x_lat, x_lat')
seRFraw3 = torch.tensor(seRFraw, dtype=torch.float).permute(3,2,1,0)
print('finish loading seRFraw', seRFraw.shape)

data = h5py.File('/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/a69_mnr1_35_nn11_20.mat', 'r')
rhomap = data['GT']['rhomap'] #(N, y_lat, depth)
rhomap3 = torch.tensor(rhomap, dtype=torch.float).permute(2,1,0)
print('finish loading rhomap', rhomap.shape)

aeration = data['GT']['aeration'] #(N,)
aeration3 = torch.tensor(aeration, dtype=torch.float).reshape(-1,)
print('finish loading aeration', aeration.shape)


# seRFraw20, rhomap20, aeration20 = load_data(Data_PATH20)
# seRFraw48, rhomap48, aeration48 = load_data(Data_PATH48)
# seRFraw69, rhomap69, aeration69 = load_data(Data_PATH69)
seRFraw = torch.cat([seRFraw1,seRFraw2,seRFraw3], dim=0)
rhomap = torch.cat([rhomap1,rhomap2,rhomap3], dim=0)
aeration = torch.cat([aeration1,aeration2,aeration3], dim=0)

torch.save(seRFraw, '/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/seRFraw.pt')
torch.save(rhomap, '/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/rhomap.pt')
torch.save(aeration, '/home/wumming/Documents/GNN-PDE/graph-pde/data/ultrasound/aeration.pt')


