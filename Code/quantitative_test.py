from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import AcaDataLoader
from model import AcaModel


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='Acamodel PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet121')
parser.add_argument('--aca_size',                  type=int,   help='initial num_filters in acamodel', default=512)

parser.add_argument('--model_name', type=str, help='model name', default='model-43500-no-aug')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)



if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__('model')).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    print(val)
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]



def online_eval(model, dataloader_eval, gpu, ngpus):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                continue

            _, _, _, _, pred_depth = model(image, focal)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()


        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)


        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1



    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None


def test(params):
    args.distributed = False
    gpu = args.gpu
    eval_measures = torch.zeros(10).cuda(device=gpu)
    args.mode = 'test'
    dataloader = AcaDataLoader(args, 'online_eval')
    
    model = AcaModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []

    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = torch.autograd.Variable(sample['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample['focal'].cuda(gpu, non_blocking=True))
            gt_depth = sample['depth']
            has_valid_depth = sample['has_valid_depth']
            if not has_valid_depth:
                continue

            _, _, _, _, pred_depth = model(image, focal)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()


        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)


        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1



    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None




    #         image = Variable(sample['image'])
    #         # image = Variable(sample['image'].cuda())
    #         focal = Variable(sample['focal'])
    #         # Predict
    #         _, _, _, _, depth_est = model(image, focal)
    #         pred_depths.append(depth_est.cpu().numpy().squeeze())

    # elapsed_time = time.time() - start_time
    # print('Elapesed time: %s' % str(elapsed_time))
    # print('Done.')
    
    # save_name = 'result_' + args.model_name
    
    # print('Saving result pngs..')
    # if not os.path.exists(os.path.dirname(save_name)):
    #     try:
    #         os.mkdir(save_name)
    #         os.mkdir(save_name + '/raw')
    #         os.mkdir(save_name + '/cmap')
    #         os.mkdir(save_name + '/rgb')
    #         os.mkdir(save_name + '/gt')
    #     except OSError as e:
    #         if e.errno != errno.EEXIST:
    #             raise
    
    # for s in tqdm(range(num_test_samples)):
    
    #     scene_name = lines[s].split()[0].split('/')[0]
    #     filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
    #         '.jpg', '.png')
    #     filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
    #         '.jpg', '.png')
    #     filename_gt_png = save_name + '/gt/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
    #         '.jpg', '.png')
    #     filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/')[1]
        
    #     rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
    #     image = cv2.imread(rgb_path)
        
    #     gt_path = os.path.join(args.data_path, './' + lines[s].split()[1])
    #     gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
    #     gt[gt == 0] = np.amax(gt)
        
    #     pred_depth = pred_depths[s]
        # pred_8x8 = pred_8x8s[s]
        # pred_4x4 = pred_4x4s[s]
        # pred_2x2 = pred_2x2s[s]
        # pred_1x1 = pred_1x1s[s]
        
        # pred_depth_scaled = pred_depth * 1000.0
        
        # pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        # cv2.imwrite(filename_pred_png+'.png', pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # if args.save_lpg:
        #     cv2.imwrite(filename_image_png+'.png', image[10:-1 - 9, 10:-1 - 9, :])

        #     plt.imsave(filename_pred_png+'_cmp.png', pred_depth_scaled, cmap=plt.colormaps['plasma'])
        #     plt.imsave(filename_gt_png+'.png', np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
        #     pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
        #     plt.imsave(filename_cmap_png+'.png', np.log10(pred_depth_cropped), cmap='plasma')
        #     pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
        #     filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
        #     plt.imsave(filename_lpg_cmap_png+'.png', np.log10(pred_8x8_cropped), cmap='Greys')
        #     pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
        #     filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
        #     plt.imsave(filename_lpg_cmap_png+'.png', np.log10(pred_4x4_cropped), cmap='Greys')
        #     pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
        #     filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
        #     plt.imsave(filename_lpg_cmap_png+'.png', np.log10(pred_2x2_cropped), cmap='Greys')
        #     pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
        #     filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
        #     plt.imsave(filename_lpg_cmap_png+'.png', np.log10(pred_1x1_cropped), cmap='Greys')
            
    
    return


if __name__ == '__main__':
    test(args)
