import os

import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import skimage.io as io
from pandas.errors import EmptyDataError


class AerialImagesDataset(Dataset):
    def __init__(self, root_csv_files, root_img_files, transform=None):

        self.bboxes = []
        self.images = []
        for csv_file in sorted(os.listdir(root_csv_files)):
            try:
                self.bboxes.append(pd.read_csv(os.path.join(root_csv_files, csv_file), header=None))
            except EmptyDataError:
                print('Image corresponding to this file has no annotations ', csv_file)
        for image in sorted(os.listdir(root_img_files)):
            self.images.append(os.path.join(root_img_files, image))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.images[idx])
        annotations = np.array(self.bboxes[idx])
        sample = {'image': image, 'annotations': annotations}

        if self.transform:
            sample = self.transform(sample)

        return sample


root_csv = input('Enter root directory for csv files: ')
root_img = input('Enter root directory for train images: ')

aerial_dataset = AerialImagesDataset(root_csv_files=root_csv, root_img_files=root_img)

fig = plt.figure()

'''
    Example with first 2 images of training dataset
'''
for i in range(2):
    sample = aerial_dataset[i]

    print(i, sample['image'].shape, sample['annotations'].shape)
