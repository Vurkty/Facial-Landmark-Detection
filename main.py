'''
Facial landmark detection with Stacked HGs or FAN Net
BUPT PRIS LAB
Vurkty
2019.3
'''

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import datetime
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data import FaceKeypointsDataset
from data import Rescale, RandomCrop, Normalize, ToTensor, RandomFlip
from config import DefaultConfig


def main():
    # Get the args.
    # Sample : "python main.py device=0 lr=0.002 epoch=20 (model=saved_model_0.pt)"
    args = sys.argv
    if len(args) < 4 or len(args) > 5 :
        print('Please input correct args!')
        exit()
    solver = Solver(args, DefaultConfig())
    solver.run()

class Solver(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None

        self.n_epochs = 0
        self.learning_rate = 0.002
        self.cuda_num = 0
        self.model = 0

    def get_args(self):
        self.cuda_num = self.args[1].split('=', -1)[1]
        self.learning_rate = self.args[2].split('=', -1)[1]
        self.n_epochs = self.args[3].split('=', -1)[1]
        if len(self.args) == 5 :
            self.model = self.args[4].split('=', -1)[1]

    def generate_heatmap(self, center, cropsize, stride = 1, sigma = 5.):
        grid = cropsize / stride
        start = stride / 2.0 - 0.5
        xyrange = [i for i in range(grid)]
        xx, yy = np.meshgrid(xyrange, xyrange)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma /sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def generate_heatmaps(self, kpts, cropsize):
        all_kpts = kpts
        batch_num = all_kpts.shape[0]
        pts_num = all_kpts.shape[1]
        heatmaps_img = []
        for j in range(batch_num):
            heatmap_img = []
            for k in range(pts_num):
                heatmap = self.generate_heatmap(all_kpts[j][k], cropsize)
                heatmap = heatmap[np.newaxis, ...]
                heatmap_img.append(heatmap)
            heatmap_img = np.concatenate(heatmap_img, axis = 0)
            heatmap_img = heatmap_img[np.newaxis, ...]
            heatmaps_img.append(heatmap_img)
        heatmaps_img = np.concatenate(heatmaps_img, axis = 0)
        heatmaps_img = Variable(torch.FloatTensor(heatmaps_img)).to(self.device)

        return heatmaps_img

    def get_pts(self, heatmaps):
        N, C, H, W = heatmaps.shape
        all_pts = []
        for i in range(N):
            pts = []
            for j in range(C):
                yy, xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
                y = yy[0]
                x = xx[0]
                pts.append([x, y])
            all_pts.append(pts)
        all_pts = np.array(all_pts)
        all_pts = Variable(torch.Tensor(all_pts), requires_grad = False)

        return all_pts

    def load_train_data(self):
        train_transform = transforms.Compose([
            Rescale(self.config.scalesize),
            RandomCrop(self.config.cropsize),
            RandomFlip(),
            Normalize(self.config.cropsize),
            ToTensor()
        ])

            # Brightness(0.7),
            # Normalize(self.config.cropsize),

        assert (train_transform is not None), 'Define a data_transform'
        train_transformed_dataset = FaceKeypointsDataset(csv_file=self.config.train_csv_path,
                                                           root_dir=self.config.train_path,
                                                           transform=train_transform)
        print('Number of train images: %d' % len(train_transformed_dataset))
        self.train_loader = DataLoader(train_transformed_dataset,
                                  batch_size=self.config.train_batch_size,
                                  shuffle=True,
                                  num_workers=4)

    def load_test_data(self):
        test_transform = transforms.Compose([
            Rescale(self.config.scalesize),
            RandomCrop(self.config.cropsize),
            Normalize(self.config.cropsize),
            ToTensor()])
        assert (test_transform is not None), 'Define a data_transform'
        test_transformed_dataset = FaceKeypointsDataset(csv_file=self.config.test_csv_path,
                                                          root_dir=self.config.test_path,
                                                          transform=test_transform)
        print('Number of test images: %d' % len(test_transformed_dataset))
        self.test_loader = DataLoader(test_transformed_dataset,
                                 batch_size=self.config.test_batch_size,
                                 shuffle=True,
                                 num_workers=4)

    def load_model(self):
        self.device = torch.device('cuda:' + self.cuda_num
                                    if torch.cuda.is_available() else 'cpu')
        print(self.device)
        from HGs import HGs
        from FAN import FAN
        self.net = HGs()

        '''
        to load the saved model
        '''
        if self.model != 0 :
            self.net.load_state_dict(torch.load(self.model))
            print('Saved model has been load.' + self.model)
        else:
            print('No model has been load.')

        self.net.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr = float(self.learning_rate))

    def save_model(self, model_dir, loss):

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        save_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
        model_name = (save_time +
                      '_lr_' + str(self.learning_rate) +
                      '_loss_' + str(loss) + '.pth'
                      )
        # Delete the last model.
        model_list = os.listdir(model_dir)
        for model in model_list:
            os.remove(model_dir + '/' + model)
            print('Remove ' + model + ' successfuly!')

        torch.save(self.net.state_dict(), model_dir + '/' + model_name)
        print('Save successfuly! ' + model_name)

    def train(self):
        self.net.train()
        train_loss = 0
        train_pts_loss = 0

        for batch_i, data in enumerate(self.train_loader):
            images = data['image']
            images = Variable(images.type(torch.FloatTensor).to(self.device))
            key_pts = data['keypoints']
            key_pts = Variable(key_pts.type(torch.FloatTensor))
            heatmaps = self.generate_heatmaps(key_pts, self.config.cropsize)

            self.optimizer.zero_grad()
            output = self.net.forward(images)
            loss = self.criterion(output, heatmaps)
            loss.backward()
            self.optimizer.step()

            output_pts = self.get_pts(output)
            pts_loss = self.criterion(key_pts, output_pts)

            #print(loss.item(), pts_loss.item() ** 0.5)

            train_loss += loss.item() * 100
            train_pts_loss += pts_loss.item() ** 0.5

            ''' 
            # You can print heatmaps here.
            for i in range(68):
                tmp_img = output_pts[-1][0, i, :, :].cpu().data.numpy()
                tmp_img = (tmp_img + 1) * 255
                cv2.imwrite('hms/' + str(batch_i) + '_' + str(i) + '.png', tmp_img)
            '''

        train_avg_loss = train_loss / len(self.train_loader)
        train_avg_pts_loss = train_pts_loss / len(self.train_loader)
        print("Training Loss:{:.3f};".format(train_avg_loss)),
        print("Training PTS Loss:{:.3f};".format(train_avg_pts_loss)),

    def test(self):
        self.net.eval()
        test_loss = 0
        test_pts_loss = 0

        for batch_it, datat in enumerate(self.test_loader):
            images = datat['image']
            images = Variable(images.type(torch.FloatTensor).to(self.device))
            key_pts = datat['keypoints']
            key_pts = Variable(key_pts.type(torch.FloatTensor))
            heatmaps = self.generate_heatmaps(key_pts, self.config.cropsize)

            output = self.net.forward(images.to(self.device))
            loss = self.criterion(output, heatmaps)

            output_pts = self.get_pts(output)
            pts_loss = self.criterion(key_pts, output_pts)

            test_loss += loss.item() * 100
            test_pts_loss += pts_loss.item() ** 0.5

        test_avg_loss = test_loss / len(self.test_loader)
        test_avg_pts_loss = test_pts_loss / len(self.test_loader)
        print("Test Loss:{:.3f};".format(test_avg_loss)),
        print("Test PTS Loss:{:.3f}...".format(test_avg_pts_loss)),

        return test_avg_pts_loss

    def run(self):
        self.get_args()
        self.load_train_data()
        self.load_test_data()
        self.load_model()
        model_dir = '3k_224_HGs_saved_model'
        best_loss = 99999.
        for epoch in range(int(self.n_epochs)):
            start_time = time.time()
            print('Epoch:%d/%d:' % (epoch + 1, int(self.n_epochs))),
            self.train()
            current_loss = self.test()
            print('Time cost : ' + str(int(time.time() - start_time)) + 's.')

            if best_loss > current_loss:
                best_loss = current_loss
                self.save_model(model_dir, best_loss)


if __name__ == '__main__':
    main()
