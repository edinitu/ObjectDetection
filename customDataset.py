import os
import sys

import torch.utils.data
import yaml
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
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

        image = plt.imread(self.images[idx])
        image = image.astype(np.float32)
        # Some images from the dataset are greyscale, so they need to be
        # converted to RGB before placing them as input in the network.
        if image.shape == (448, 448):
            image = utils.grey2rgb(image)

        # Retain only RGB values from images with an Alpha channel
        if image.shape == (448, 448, 4):
            image = image[:, :, 0:3]

        # Normalize image to have pixel values in [0,1] interval
        if np.max(image) - np.min(image) != 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        else:
            # if image cannot be normalized, something is wrong, choose another random image
            index = np.random.randint(0, len(self.images))
            image, grid_annotations = self.__getitem__(index)
            return image, grid_annotations

        if np.min(image) < 0 or np.max(image) > 1:
            raise RuntimeError(f'Image values out of range: max {np.max(image)}, min {np.min(image)}')
        annotations = np.array(self.annotations[idx])
        # get the vectors for every grid cell in the image
        grids_annotations = self.build_grids_annotations(annotations)
        grids_annotations = grids_annotations.astype(np.float32)
        #sample = {'image': image, 'annotations': grids_annotations}

        if self.transform:
            #sample = self.transform(sample)
            image = self.transform(image)
            grids_annotations = self.transform(grids_annotations)

        return image, grids_annotations

    def build_grids_annotations(self, annotations):
        # grid cell dimension, currently just for 7x7 grid
        # TODO add grid dimensions to dataset-config and make it class field.
        grid_dim = int(self.img_dim/7)
        img_ground_truth = []
        # Loop through grid cells
        for i in range(7):
            for j in range(7):
                # Ground truth for a grid cell is one bounding box (IOU, x, y, w, h) with its class
                # probabilities for TWO objects. Initialize it with confidence 0 for both objects and
                # random coordinates and set it accordingly if a grid cell contains one or two objects.
                grid_vector = np.random.rand((5 + self.no_of_classes)*2, 1)
                # before checking, assume no object is in the cell
                grid_vector[0] = 0
                grid_vector[21] = 0
                objects_in_cell = 0
                # if a bbox center is in the grid cell => grid cell responsible for predicting that
                # object
                for annt in annotations:
                    bbox = annt[1:]
                    if i * grid_dim <= bbox[0] <= (i + 1) * grid_dim and \
                            j * grid_dim <= bbox[1] <= (j + 1) * grid_dim and \
                            objects_in_cell < 2:
                        self.build_grid_vector(grid_vector, bbox, grid_dim, annt[0], objects_in_cell, i, j)
                        objects_in_cell += 1

                img_ground_truth.append(grid_vector)

        return np.array(img_ground_truth)

    def build_grid_vector(self, grid_vector, bbox, grid_dim, class_id, objects_in_cell, i, j):
        # Objects in cell can only be 0 or 1. If it's 0, then we set the first part of the grid
        # vector (first 21 elements), else we set the second part (for the second object in that
        # grid).

        # We have an object in this grid cell => P(Obj) = 1.
        grid_vector[21*objects_in_cell + 0] = 1
        # center is an offset in the grid cell
        grid_vector[21*objects_in_cell + 1] = bbox[0] / ((i + 1) * grid_dim)
        grid_vector[21*objects_in_cell + 2] = bbox[1] / ((j + 1) * grid_dim)
        # normalize w and h by dividing them to img width and height
        grid_vector[21*objects_in_cell + 3] = bbox[2] / self.img_dim
        grid_vector[21*objects_in_cell + 4] = bbox[3] / self.img_dim
        # set to 1 the class id, others with 0
        for c in range(self.no_of_classes):
            if class_id == c:
                grid_vector[21*objects_in_cell + 5+c] = 1
            else:
                grid_vector[21*objects_in_cell + 5+c] = 0


def example():
    """
        Example with first 2 images of training dataset.
        Print images shape and vectors for grid cells that
        contain 2 objects.
    """
    with open('configs/dataset-config.yml') as f:
        paths = yaml.safe_load(f)

    root_csv = paths['train_labels_csv']
    root_img = paths['train_images_path']
    dim = paths['img_dim']
    classes = paths['no_of_classes']

    aerial_dataset = AerialImagesDataset(root_csv_files=root_csv, root_img_files=root_img
                                         , img_dim=dim, no_of_classes=classes)
    for i in range(2):
        img, annt = aerial_dataset[i]

        print(i, img.shape)
        for elem in annt:
            if elem[0] == 1 and elem[21] == 1:
                print(elem)


#example()
