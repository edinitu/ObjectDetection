import os

import torch.utils.data
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import skimage.io as io
from pandas.errors import EmptyDataError


class AerialImagesDataset(Dataset):
    def __init__(self, root_csv_files, root_img_files, img_dim, no_of_classes, transform=None):
        self.img_dim = img_dim
        self.no_of_classes = no_of_classes
        self.annotations = []
        self.images = []
        for csv_file in sorted(os.listdir(root_csv_files)):
            try:
                self.annotations.append(pd.read_csv(os.path.join(root_csv_files, csv_file), header=None))
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
        annotations = np.array(self.annotations[idx])
        grids_annotations = self.build_grids_annotations(annotations)
        sample = {'image': image, 'annotations': grids_annotations}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def build_grids_annotations(self, annotations):
        dim = int(self.img_dim/7)
        img_ground_truth = []
        for i in range(7):
            for j in range(7):
                # Ground truth for a grid cell is one bounding box (IOU, x, y, w, h) with its class
                # probabilities. Initialize it with confidence 0 and random coordinates and set it
                # accordingly if a grid cell contains an object.
                grid_vector = np.random.rand(21, 1)
                grid_vector[0] = 0
                for annt in annotations:
                    bbox = annt[1:]
                    if i * dim <= bbox[0] <= (i + 1) * dim and \
                            j * dim <= bbox[1] <= (j + 1) * dim:
                        # we have an object in this grid cell => P(Obj) = 1
                        grid_vector[0] = 1
                        # center is an offset in the grid cell
                        grid_vector[1] = bbox[0] / ((i + 1) * dim)
                        grid_vector[2] = bbox[1] / ((j + 1) * dim)
                        # normalize w and h by dividing them to img width and height
                        grid_vector[3] = bbox[2] / self.img_dim
                        grid_vector[4] = bbox[3] / self.img_dim
                        self.add_class_probabilities(grid_vector, annt[0])
                img_ground_truth.append(grid_vector)

        return np.array(img_ground_truth)

    def add_class_probabilities(self, grid_vector, class_id):
        for i in range(self.no_of_classes):
            if class_id == i:
                grid_vector[5+i] = 1
            else:
                grid_vector[5+i] = 0


with open('configs/dataset-config.yml') as f:
    paths = yaml.safe_load(f)

root_csv = paths['train_labels_csv']
root_img = paths['train_images_path']
dim = paths['img_dim']
classes = paths['no_of_classes']

aerial_dataset = AerialImagesDataset(root_csv_files=root_csv, root_img_files=root_img
                                     , img_dim=dim, no_of_classes=classes)

fig = plt.figure()

'''
    Example with first 2 images of training dataset
'''
for i in range(2):
    sample = aerial_dataset[i]

    print(i, sample['image'].shape)
    for elem in sample['annotations']:
        if elem[0] == 1:
            print(elem)
