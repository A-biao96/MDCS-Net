from torch import nn

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math

# Reshape + Concat layer

class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0

    def __init__(self, block_size):
        # super(Reshape_Concat_Adap, self).__init__()
        Reshape_Concat_Adap.blocksize = block_size

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)

        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize), int(h_ * Reshape_Concat_Adap.blocksize))).cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                # data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                # print data_temp.shape
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)



class ResidualBlock(nn.Module):
    def __init__(self, channels, has_BN=None):
        super(ResidualBlock, self).__init__()
        self.has_BN = has_BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        if has_BN:
            self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        # self.relu1 = nn.ReLU()
        if has_BN:
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        if self.has_BN:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.has_BN:
            out = self.bn2(out)
        return out

class RFANet1(nn.Module):
    def __init__(self, blocksize=32, subrate=0.3):
        super(RFANet1, self).__init__()
        self.blocksize = blocksize

        # for sampling
        self.sampling = nn.Conv2d(4, int(np.round(blocksize * blocksize * subrate)), kernel_size=blocksize, stride=blocksize,
                                  padding=0, bias=False)
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), blocksize * blocksize, kernel_size=1, stride=1,
                                    padding=0,bias=False)

        self.Conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.Conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2, stride=1, bias=False)
        self.Conv3 = nn.Conv2d(1, 1, kernel_size=7, padding=3, stride=1, bias=False)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )
        self.Conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1, bias=True),
                             nn.ReLU())
        self.Conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=True),
                         nn.ReLU())
        self.block2 = ResidualBlock(64, has_BN=False)

        self.block3 = ResidualBlock(64, has_BN=False)

        self.block4 = ResidualBlock(64, has_BN=False)

        self.block5 = ResidualBlock(64, has_BN=False)

        self.block6 = nn.Conv2d(256, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.block7 = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, x):
        Conv1 = self.Conv1(x)
        Conv2 = self.Conv2(x)
        Conv3 = self.Conv3(x)
        add1 = Conv1 + x
        add2 = Conv1 + Conv2
        add3 = Conv2 + Conv3
        concat1 = torch.cat([add1, add2, add3, Conv3], dim=1)
        x1 = self.sampling(concat1)


        x1 = self.upsampling(x1)
        x1 = My_Reshape_Adap(x1, self.blocksize)  # Reshape + Concat

        block1 = self.block1(x1)
        Conv4 = self.Conv4(block1)
        Conv5 = self.Conv5(Conv4)
        block2 = self.block2(Conv5)

        add1 = block2 + Conv5
        block3 = self.block3(add1)

        add2 = add1 + block3
        block4 = self.block4(add2)

        add3 = add2 + block4
        block5 = self.block5(add3)

        concat = torch.cat([block2, block3, block4, block5], dim=1)
        block6 = self.block6(concat)

        block7 = self.block7(block6)
        out = x1 + block7
        return out




if __name__ == '__main__':
    import torch

    img = torch.randn(1, 1, 32, 32)
    net =RFANet1()
    out = net(img)
    print(out.size())