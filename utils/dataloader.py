from random import shuffle
from PIL import Image
import copy
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch

from core.datasets.compose import Compose

class Mydataset(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):
        
        return len(self.gt_labels)

    # def __getitem__(self, index):
    #     image_path = self.gt_labels[index].split(' ')[0].split()[0]
    #     image = Image.open(image_path)
    #     cfg = copy.deepcopy(self.cfg)
    #     image = self.preprocess(image,cfg)
    #     gt = int(self.gt_labels[index].split(' ')[1])
        
    #     return image, gt, image_path
    
    # def preprocess(self, image,cfg):
    #     if not (len(np.shape(image)) == 3 and np.shape(image)[2] == 3):
    #         image = image.convert('RGB')
    #     funcs = []

    #     for func in cfg:
    #         funcs.append(eval('transforms.'+func.pop('type'))(**func))
    #     image = transforms.Compose(funcs)(image)
    #     return image

    def __getitem__(self, index):
        results = self.pipeline(copy.deepcopy(self.data_infos[index]))
        return results['img'], int(results['gt_label']), results['filename']
    
    def load_annotations(self):
        """Load image paths and gt_labels."""
        if len(self.gt_labels) == 0:
            raise TypeError('ann_file is None')
        samples = [x.strip().rsplit(' ', 1) for x in self.gt_labels]
        
        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
    
    def get_labels(self):
        return self.gt_labels

class Mymixdataset(Dataset):
    def __init__(self, real_data, aug_data_dir, cfg, aug_transform):
        self.cfg = cfg

        self.real_data = real_data
        self.real_pipeline = Compose(self.cfg)
        self.real_data_infos = self.load_annotations()
        
        self.aug_data_infos = np.load(aug_data_dir)
        self.aug_pipeline = Compose(aug_transform)
        print(self.aug_data_infos['arr_0'].shape)

        
    def __len__(self):
        return int(len(self.real_data))+int(self.aug_data_infos['arr_0'].shape[0])
        
    def __getitem__(self, index):
        #print(index)
        if index < len(self.real_data):
            #print("index within 10015:",index)
            results = self.real_pipeline(copy.deepcopy(self.real_data_infos[index]))
            temp_img = Image.open(self.real_data_infos[index]['img_info']['filename'])
            #print(index, temp_img.getextrema())
            #print(results['img'].size(), results['img'].min(), results['img'].max())
            return results['img'], int(results['gt_label']), results['filename']
        else:
            #print("index out 10015:",index-10015)
            index = index-len(self.real_data)
            img = self.aug_data_infos['arr_0'][int(index)]
            pil_image = Image.fromarray(img)
            #print(index, pil_image.getextrema())
            results_img = self.aug_pipeline(copy.deepcopy(pil_image))
            results_label = self.aug_data_infos['arr_1'][int(index)]
            #print(results_img.size(), results_img.min(), results_img.max())
            return results_img, int(results_label), str(index)
    
    def load_annotations(self):
        """Load image paths and gt_labels."""
        if len(self.real_data) == 0:
            raise TypeError('ann_file is None')
        samples = [x.strip().rsplit(' ', 1) for x in self.real_data]
        
        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
    
    def get_labels(self):
        return self.real_data

def collate(batches):
    images, gts, image_path = tuple(zip(*batches))
    images = torch.stack(images, dim=0)
    gts = torch.as_tensor(gts)
    
    return images, gts, image_path

class Myaugdataset(Dataset):
    def __init__(self, aug_data_dir, aug_transform, num_classes=7):
        #self.cfg = cfg

        #self.real_data = real_data
        #self.real_pipeline = Compose(self.cfg)
        #self.real_data_infos = self.load_annotations()
        
        self.aug_data_infos = np.load(aug_data_dir)
        self.aug_pipeline = Compose(aug_transform)
        print(self.aug_data_infos['arr_0'].shape)

        self.num_classes = num_classes

        
    def __len__(self):
        return int(self.aug_data_infos['arr_0'].shape[0])
        
    def __getitem__(self, index):
        #print(index)
        
        #print("index out 10015:",index-10015)
        #index = index-2931
        img = self.aug_data_infos['arr_0'][int(index)]
        pil_image = Image.fromarray(img)
        #print(index, pil_image.getextrema())
        results_img = self.aug_pipeline(copy.deepcopy(pil_image))
        results_label = self.aug_data_infos['arr_1'][int(index)]
        #print(results_img.size(), results_img.min(), results_img.max())
        return results_img, int(results_label), str(index)
    
    def get_labels(self):
        return self.real_data
