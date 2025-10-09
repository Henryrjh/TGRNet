
import argparse
import time
import numpy as np
import torch
import pandas as pd
import cv2
from TGRNet.TGRNet import create_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_rs_epe(flow_gt, flow_pr, valid_gt):
    # 将输入数据转换为 64 位浮点数进行计算
    flow_gt = flow_gt.astype(np.float64)
    flow_pr = flow_pr.astype(np.float64)
    valid_gt = valid_gt.astype(np.float64)

    error = np.sum(np.abs(flow_gt - flow_pr) * valid_gt)

    nums = np.sum(valid_gt)

    epe = error / nums

    return error, nums, epe

def compute_rs_d1(flow_gt, flow_pr, valid_gt, thresold = 3):

    # 将输入数据转换为 64 位浮点数进行计算
    flow_gt = flow_gt.astype(np.float64)
    flow_pr = flow_pr.astype(np.float64)
    valid_gt = valid_gt.astype(np.float64)

    err_map = np.abs(flow_gt - flow_pr) * valid_gt

    err_mask = err_map > thresold

    err_disps = np.sum(err_mask.astype('float64'))

    nums = np.sum(valid_gt)

    d1 = err_disps / nums

    return err_disps, nums, d1

def pair_read_fromfolder(folders_list, keys_list, names_list):

    assert len(folders_list) == len(keys_list)

    data = []

    for filename in os.listdir(folders_list[0]):

        temp_dict = {}

        for folder, key, name in zip(folders_list, keys_list, names_list):

            temp_filename = filename.replace(names_list[0], name)
            filepath = os.path.join(folder, temp_filename)
            
            temp_dict[f'{key}_path'] = filepath
        
        data.append(temp_dict)
    
    return data

def img_norm(img):
    mean = np.mean(img)
    std = np.std(img)
    new_img = (img - mean) / std
    return new_img

@torch.no_grad()
def validate_rs(model, left_dir, right_dir, disp_dir, mode='16bit', device='cuda', 
                max_disp=64, min_disp=-128, save_path=None, dataset_name='whu', mixed_prec=False):
    """ Peform validation using the rs split """

    if save_path is not None:

        os.makedirs(save_path, exist_ok=True)

        visual_dir = os.path.join(save_path, 'visuals')
        os.makedirs(visual_dir, exist_ok=True)

        csv_path = os.path.join(save_path, f'TGRNet_results_{dataset_name}.csv')

    model.eval()
    model.to(device)

    total_epe_nums = 0
    total_d1_nums = 0
    total_epe_error = 0
    total_err_disps = 0
    total_time = 0

    metric_list = []

    all_files_paths = pair_read_fromfolder([left_dir, right_dir, disp_dir], ['left', 'right', 'left_disparity'], ['left', 'right', 'disparity'])

    for val_id in range(len(all_files_paths)):

        left_path = all_files_paths[val_id]['left_path']
        right_path = all_files_paths[val_id]['right_path']
        disp_gt_path = all_files_paths[val_id]['left_disparity_path']

        left_img = cv2.imread(left_path, -1)
        right_img = cv2.imread(right_path, -1)
        disp_gt = cv2.imread(disp_gt_path, -1)
        valid_gt =  (disp_gt < max_disp) & (disp_gt > min_disp)

        if mode == '16bit': ### WHU        
            assert (len(left_img.shape)==2), '16bit must be single channel!'
            left_img = np.tile(left_img[...,None], (1, 1, 3))
            right_img = np.tile(right_img[...,None], (1, 1, 3))
            left_img = img_norm(left_img)
            right_img = img_norm(right_img)
        elif mode == '8bit':    ### US3D
            assert (len(left_img.shape)==3) and (left_img.shape[2]==3), '8bit must be 3 channel!'
            left_img = left_img[..., :3]
            right_img = right_img[..., :3]
            left_img = left_img.astype(np.float32) / 255.
            right_img = right_img.astype(np.float32) / 255. 
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float()
        right_img = torch.from_numpy(right_img).permute(2, 0, 1).float()

        image1 = left_img[None].to(device)
        image2 = right_img[None].to(device)
        with autocast(enabled=mixed_prec):
            start_time = time.time()
            _, flow_pr = model(image1, image2, test_mode=True)
            end_time = time.time()
            run_time = end_time - start_time

        flow_pr = flow_pr.float().cpu().squeeze(0)
        pre_disp = - np.array(flow_pr[0])

        epe_error, epe_nums, epe = compute_rs_epe(disp_gt, pre_disp, valid_gt)
        err_disps, d1_nums, d1 = compute_rs_d1(disp_gt, pre_disp, valid_gt, 3)

        disp_filename = (os.path.splitext(os.path.split(left_path)[1])[0]).replace('disparity', 'pred')

        if save_path is not None:
            cv2.imwrite(os.path.join(visual_dir, disp_filename + '.tif'), pre_disp)

        metric_dict = {
            'filename': disp_filename,
            'epe': round(epe,4), 'd1_3px': round(d1,4), 'time': round(run_time, 4)
        }
        metric_list.append(metric_dict)

        total_epe_error += epe_error
        total_epe_nums += epe_nums
        total_err_disps += err_disps
        total_d1_nums += d1_nums
        total_time += run_time

        print(f"RS {disp_filename} {val_id+1}-th out of {len(all_files_paths)}. EPE {round(epe,4)} D1 {round(d1,4)} time {round(run_time,4)}")

    total_epe = total_epe_error / total_epe_nums

    total_d1 = total_err_disps / total_d1_nums

    avg_time = total_time / len(all_files_paths)

    if save_path is not None:

        avg_dict = {
            'filename': 'Total',
            'epe': round(total_epe,4), 'd1_3px': round(total_d1,4), 'time': avg_time,
        }

        metric_list.append(avg_dict)

        metric_df = pd.DataFrame(metric_list)
        metric_df.to_csv(csv_path, index=False)

    print("Validation WHU: EPE %f, D1 %f, time: %f" % (total_epe, total_d1, avg_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='16bit', choices=['8bit', '16bit'], help='The format of input data')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='cpu or cuda')

    parser.add_argument('--test_left_dir', default=f'dataset/WHU-Stereo/with_GT/test_all/left', help="")
    parser.add_argument('--test_right_dir', default=f'dataset/WHU-Stereo/test_all/right', help="")
    parser.add_argument('--test_disp_dir', default=f'dataset/WHU-Stereo/test_all/disp', help="")
    parser.add_argument('--max_disp', default=64, help="max_disp")
    parser.add_argument('--min_disp', default=-128, help="min_disp")
    parser.add_argument('--test_save_path', default=f'results/whu', help="The dir you want to save")    

    args = parser.parse_args()

    model = create_model()
    weight_path = 'weights/new_rs_whu.pth'
    weight = torch.load(weight_path)['model']
    model.load_state_dict(weight, strict=True)

    validate_rs(model, left_dir=args.test_left_dir, right_dir=args.test_right_dir, disp_dir=args.test_disp_dir,
                save_path = args.test_save_path, device=args.device, mode=args.mode,
                max_disp=args.max_disp, min_disp=args.min_disp)