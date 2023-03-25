import os.path
import sys
import time
import numpy as np
import torch
import yaml
import customDataset as dataset
import torchvision.transforms as tv
from torch.utils.data import DataLoader
import utils
from metrics import AveragePrecision
import pandas as pd
import matplotlib.pyplot as plt
from model import NetworkModel

img_test_path: str
csv_test_path: str
dim: int
no_of_classes: int
weights: dict
obj_in_grid: int
one_image: bool
draw_ground_truth: bool
image_path: str
labels: dict


def init():
    with open('configs/model-config.yaml') as f:
        configs = yaml.safe_load(f)

    testing_cfg = configs['testing']
    general_cfg = configs['general']
    global img_test_path
    img_test_path = testing_cfg['testing_img']
    global csv_test_path
    csv_test_path = testing_cfg['testing_csv']
    global dim
    dim = general_cfg['img_dim']
    global no_of_classes
    no_of_classes = general_cfg['no_of_classes']
    global weights
    weights = torch.load(testing_cfg['weights'])
    global obj_in_grid
    obj_in_grid = general_cfg['objects_in_grid']
    global one_image
    one_image = testing_cfg['oneImage']
    if one_image:
        global image_path
        image_path = testing_cfg['image']
        global draw_ground_truth
        draw_ground_truth = testing_cfg['draw_ground_truth']

    with open('configs/pre-processing-config.yaml') as f:
        preproc_config = yaml.safe_load(f)

    global labels
    labels = preproc_config['processImages']['labels']


if __name__ == "__main__":
    init()

    # Loading saved model
    network = NetworkModel(no_of_classes, obj_in_grid, testing=True)
    try:
        network.load_state_dict(weights)
    except RuntimeError:
        print('Weights from file don\'t match model\'s weights shape, please check number of classes'
              ' and number of objects to be detected in a grid cell')
        print('exiting...')
        time.sleep(2)
        sys.exit(-1)

    network.eval()
    network.cuda()

    if one_image:
        annotations = pd.read_csv(os.path.join(csv_test_path, image_path + '.csv'), header=None)
        annotations = np.array(annotations)
        image = plt.imread(os.path.join(img_test_path, image_path + '.png'))
        image = image.astype(np.float16)
        # Some images from the dataset are greyscale, so they need to be
        # converted to RGB before placing them as input in the network.
        if image.shape == (dim, dim):
            image = utils.grey2rgb(image)

        # Retain only RGB values from images with an Alpha channel
        if image.shape == (dim, dim, 4):
            image = image[:, :, 0:3]

        # Normalize image to have pixel values in [0,1] interval
        if np.max(image) - np.min(image) != 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

        plt.figure()
        transform = tv.Compose([tv.ToTensor()])
        image = transform(image)
        image = torch.reshape(image, (1, 3, dim, -1))
        annotations = dataset.AerialImagesDataset.no_args_construct().build_grids_annotations(annotations)
        annotations = annotations.astype(np.float16)
        annotations = transform(annotations)
        image = image.to(torch.device('cuda'))
        annotations = annotations.reshape(1, 49 * (5 + no_of_classes))

        with torch.no_grad():
            start = time.time_ns()
            outputs = network(image)
            end = time.time_ns()

        inference_time = (end - start) * (10**(-6))
        print(f'Inference time: {inference_time}')
        print('Detected objects: ')
        final_pred = utils.FinalPredictions(outputs.cpu(), annotations)
        annt_test = utils.FinalPredictions(annotations, annotations)
        image = plt.imread(os.path.join(img_test_path, image_path + '.png'))
        final_pred.draw_boxes()
        if draw_ground_truth:
            annt_test.draw_boxes(truths=True)
        plt.imshow(image)
        plt.show(block=True)

    else:
        print('Loading the testing dataset...')
        transform = tv.Compose([tv.ToTensor()])
        testing_dataset = dataset.AerialImagesDataset(
            csv_test_path, img_test_path, dim, no_of_classes, obj_in_grid, transform=transform)
        print('Dataset ready')

        print('Loading the testing dataloader...')
        test_loader = DataLoader(dataset=testing_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('Testing dataloader ready')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        durations = []
        for i, (image, annotations) in enumerate(test_loader):
            image = image.to(device)
            annotations = annotations.reshape(-1, 49 * (5 + no_of_classes))
            with torch.no_grad():
                start = time.time_ns()
                outputs = network(image)
                end = time.time_ns()
            duration_ms = (end-start) * (10**(-6))
            durations.append(duration_ms)
            final_pred = utils.FinalPredictions(outputs.cpu().to(torch.float32), annotations.to(torch.float32))

        mAP = utils.get_mAP(labels)
        print(f'mAP: {mAP}')

        avg_inf_time = np.sum(np.asarray(durations)) / len(durations)
        print(f'Average inference time: {avg_inf_time} ms')
