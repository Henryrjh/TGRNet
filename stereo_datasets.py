
import numpy as np
import torch
import torch.utils.data as data
import logging
import os
import random

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

import cv2
class RSDataset(data.Dataset):
    def __init__(self, left, right, left_disparity, training, max_disp, min_disp=0 ,gt_size = 1024):
        self.training = training
        self.is_test = False

        self.paths = pair_read_fromfolder([left, right, left_disparity], ['left', 'right', 'left_disparity'], ['left', 'right', 'disparity'])
        self.gt_size = gt_size
        self.max_disp = max_disp
        self.min_disp = min_disp
        self.init_seed = False        

    def __getitem__(self, index):

        if self.is_test:
            left_img = cv2.imread(self.paths[index]['left_path'], -1)
            right_img = cv2.imread(self.paths[index]['right_path'], -1)
            disp = cv2.imread(self.paths[index]['left_disparity_path'], -1)
            if len(left_img.shape) == 2:
                left_img = np.tile(left_img[...,None], (1, 1, 3))
                right_img = np.tile(right_img[...,None], (1, 1, 3))
            else:
                left_img = left_img[..., :3]
                right_img = right_img[..., :3]

            left_img = img_norm(left_img)
            right_img = img_norm(right_img)
            left_img = torch.from_numpy(left_img).permute(2, 0, 1).float()
            right_img = torch.from_numpy(right_img).permute(2, 0, 1).float() 

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.paths)
        left_img = cv2.imread(self.paths[index]['left_path'], -1)
        right_img = cv2.imread(self.paths[index]['right_path'], -1)
        disp = cv2.imread(self.paths[index]['left_disparity_path'], -1)

        disp = np.array(disp).astype(np.float32)
        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        if self.training:
            
            h, w = left_img.shape[:2]

            h0 = random.randint(0, h-self.gt_size)
            w0 = random.randint(0, w-self.gt_size)

            left_img = left_img[h0: h0 + self.gt_size, w0: w0 + self.gt_size].astype('float32')
            right_img = right_img[h0: h0 + self.gt_size, w0: w0 + self.gt_size].astype('float32')
            disp = disp[h0: h0 + self.gt_size, w0: w0 + self.gt_size]

            flow = flow[h0: h0 + self.gt_size, w0: w0 + self.gt_size]

        if len(left_img.shape) == 2:
            left_img = np.tile(left_img[...,None], (1, 1, 3))
            right_img = np.tile(right_img[...,None], (1, 1, 3))
            left_img = img_norm(left_img)
            right_img = img_norm(right_img)
        else: 
            left_img = left_img[..., :3]
            right_img = right_img[..., :3]
            left_img = left_img.astype(np.float32) / 255.
            right_img = right_img.astype(np.float32) / 255. 
          

        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float()
        right_img = torch.from_numpy(right_img).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        valid = (flow[0] < -self.min_disp) & (flow[0] > -self.max_disp) & (flow[1].abs() < 128) 

        flow = flow[:1]

        path_list = [self.paths[index]['left_path']] + [self.paths[index]['right_path']] + [self.paths[index]['left_disparity_path']]

        return path_list, left_img, right_img, flow, valid.float()
                  
    def __len__(self):
        return len(self.paths)

def fetch_dataloader(args):
    train_dataset = None
    for dataset_name in args.train_datasets:       
        new_dataset = RSDataset(args.left_dir, args.right_dir, args.disp_dir,
                                args.training, args.max_disp, args.min_disp,
                                args.gt_size)

        logging.info(f"Adding {len(new_dataset)} samples from RS_Dataset")        
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

