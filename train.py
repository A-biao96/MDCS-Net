import torch
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from RFA import RFANet1
from torch import nn
import time
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import scipy.io as sio
from data_utils import TrainDatasetFromFolder
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--pre_epochs', default=100, type=int, help='pre train epoch number')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

parser.add_argument('--batchSize', default=64, type=int, help='train batch size')
parser.add_argument('--sub_rate', default=0.3, type=float, help='sampling sub rate')

parser.add_argument('--loadEpoch', default=24, type=int, help='load epoch number')
parser.add_argument('--generatorWeights', type=str,
                default= ' ',
                help="path to CSNet weights (to continue training)")
# parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
opt = parser.parse_args()

CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
NUM_EPOCHS = opt.num_epochs
PRE_EPOCHS = opt.pre_epochs
LOAD_EPOCH = 1


train_set = TrainDatasetFromFolder(' 填入路径 ', crop_size=CROP_SIZE, blocksize=BLOCK_SIZE)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)
# rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
#                              shuffle=True)
net = RFANet1(BLOCK_SIZE, opt.sub_rate)

mse_loss = nn.MSELoss()

if opt.generatorWeights != '':
       net.load_state_dict(torch.load(opt.generatorWeights))
       LOAD_EPOCH = opt.loadEpoch

if torch.cuda.is_available():
    net.cuda()
    mse_loss.cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[18, 39], gamma=0.1)  # StepLR, step_size=50

for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0, }

    net.train()

    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        running_results['batch_sizes'] += batch_size

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = net(z)
        optimizer.zero_grad()
        g_loss = mse_loss(fake_img, real_img)

        g_loss.backward()
        optimizer.step()

        running_results['g_loss'] += g_loss.item() * batch_size

        train_bar.set_description(desc='[%d] Loss_G: %.7f lr: %.7f' % (
            epoch, running_results['g_loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))


    scheduler.step()



    # for saving model
    save_dir = 'RFA1-epochs' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch % 1 == 0:
        torch.save(net.state_dict(), save_dir + '/net_epoch_%d_%6f.pth' % (epoch, running_results['g_loss']/running_results['batch_sizes']))
