import argparse
import os

import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from RFA import RFANet1
from skimage.metrics import structural_similarity as ssim
import cv2

parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model",
        default="   ",
        type=str, help="model path")
parser.add_argument("--dataset", default="Test/Set14_mat", type=str, help="dataset name, Default: Set11")
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--sub_rate', default=0.3, type=float, help='sampling sub rate')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
# parser.add_argument('--test_name', type=str, default='uban100', help='name of test set')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = RFANet1(opt.block_size, opt.sub_rate)

if opt.model != '':
    model.load_state_dict(torch.load(opt.model))

image_list = glob.glob(opt.dataset + "/*.*")

avg_psnr_predicted = 0.0
avg_elapsed_time = 0.0
avg_ssim_predicted = 0.0
for image_name in image_list:
    print("Processing ", image_name)
    im_gt_y = sio.loadmat(image_name)['im_gt_y']

    im_gt_y = im_gt_y.astype(float)

    im_input = im_gt_y / 255.

    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

    # if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
    # else:
    #     model = model.cpu()

    start_time = time.time()
    res = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    res = res.cpu()

    im_res_y = res.data[0].numpy().astype(np.float32)

    im_res_y = im_res_y * 255.
    im_res_y[im_res_y < 0] = 0
    im_res_y[im_res_y > 255.] = 255.
    im_res_y = im_res_y[0, :, :]

    psnr_predicted = PSNR(im_gt_y, im_res_y, shave_border=0)
    SSIM_predicted = ssim(im_gt_y, im_res_y, data_range=255)


    print(psnr_predicted)
    print(SSIM_predicted)
    avg_psnr_predicted += psnr_predicted
    avg_ssim_predicted += SSIM_predicted


print("Dataset=", opt.dataset)
print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
print("SSIM_predicted=", avg_ssim_predicted / len(image_list))
print("It takes average {}s for processing".format(avg_elapsed_time / len(image_list)))
