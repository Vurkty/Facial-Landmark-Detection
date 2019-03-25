import os
import torch
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2

from torch.utils.data import Dataset
from sklearn.utils import shuffle


class FaceKeypointsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.key_pts_frame = shuffle(pd.read_csv(csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)
        if (len(image.shape)==2):
            image = np.stack((image,)*3, -1)

        if(image.shape[2] == 4):#remove alpha
            image = image[:,:,0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __init__(self, cropsize, rgb=False):
        assert isinstance(cropsize, int)
        self.cropsize = cropsize

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        key_pts_copy = np.copy(key_pts)
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_copy=  image_copy/255.0
        s = self.cropsize / 2
        # key_pts_copy = (key_pts_copy - s)/s

        return {'image': image_copy, 'keypoints': key_pts_copy}

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        key_pts = sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        for i in range(len(key_pts)):
            if i % 2 == 0:
                key_pts[i] = key_pts[i] * new_w / w
            else:
                key_pts[i] = key_pts[i] * new_h / h

        return {'image': img, 'keypoints': key_pts}

class RandomCrop(object):
    def __init__(self, output_size, random_flip=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top_max = min(max(key_pts[:,1].max() - new_h, 0), h - new_h - 1)
        left_max = min(max(key_pts[:,0].max() - new_w, 0), w - new_w - 1)
        top = np.random.randint(top_max, h - new_h)
        left = np.random.randint(left_max, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        #key_pts = key_pts - [left, top]
        key_pts[:, 1] = key_pts[:, 1] - top
        key_pts[:, 0] = key_pts[:, 0] - left

        return {'image': image, 'keypoints': key_pts}

class ToTensor(object):

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).double(),
                'keypoints': torch.from_numpy(key_pts).double()}

class RandomFlip(object):

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w,_ = image.shape
        if np.random.choice((True, False)):
            image =cv2.flip(image,1)

            key_pts[:,0]=w-key_pts[:,0]
            pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10],
                 [7, 9], [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [36, 45],
                 [37, 44], [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34],
                 [50, 52], [49, 53], [48, 54], [61, 63], [60, 64], [67, 65], [59, 55], [58, 56]]

            for matched_p in pairs:

                idx1, idx2 = matched_p[0], matched_p[1]
                tmp = np.copy(key_pts[idx1])
                key_pts[idx1] =np.copy(key_pts[idx2])
                key_pts[idx2] =tmp

        return {'image': image, 'keypoints': key_pts}

class Brightness(object):

    def __init__(self, var=0.8):
        self.var = var

    def __call__(self, sample):
        image1, key_pts = sample['image'], sample['keypoints']

        if image1.any()<1:
            image1=image1*255

        image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2HSV)
        image1 = np.array(image1, dtype = np.float64)
        random_bright = np.random.uniform(low=self.var, high=1.2)

        image1[:,:,2] = (image1[:,:,2]*random_bright)
        image1[:,:,2][image1[:,:,2]>255]  = 255
        image1[:,:,2][image1[:,:,2]<0]  = 0

        image1 = np.array(image1, dtype = np.uint8)
        image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

        return {'image': image1, 'keypoints': key_pts}
