import os.path
import sys

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

if __name__ == "__main__":
    # TODO Refactor the initialization here
    with open('configs/model-config.yaml') as f:
        testing_configs = yaml.safe_load(f)

    # Loading saved model
    weights = torch.load(testing_configs['weights'])
    network = NetworkModel(testing=True)
    network.load_state_dict(weights)
    network.eval()
    network.cuda()

    img_test_path = testing_configs['testing_img']
    csv_test_path = testing_configs['testing_csv']
    dim = testing_configs['dim']
    classes = testing_configs['classes']

    if testing_configs['oneImage']:
        annotations = pd.read_csv(os.path.join(csv_test_path, testing_configs['image'] + '.csv'), header=None)
        annotations = np.array(annotations)
        #annotations = np.zeros((1, 6))
        image = plt.imread(os.path.join(img_test_path, testing_configs['image'] + '.png'))
        image = image.astype(np.float16)
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

        plt.figure()
        transform = tv.Compose([tv.ToTensor()])
        image = transform(image)
        image = torch.reshape(image, (1, 3, 448, -1))
        annotations = dataset.AerialImagesDataset.no_args_construct().build_grids_annotations(annotations)
        annotations = annotations.astype(np.float16)
        annotations = transform(annotations)
        image = image.to(torch.device('cuda'))
        annotations = annotations.reshape(1, 49 * 9)
        with torch.no_grad():
            outputs = network(image)

        final_pred = utils.FinalPredictions(outputs.cpu(), annotations)
        annt_test = utils.FinalPredictions(annotations, annotations)
        image = plt.imread(os.path.join(img_test_path, testing_configs['image'] + '.png'))
        final_pred.draw_boxes()
        annt_test.draw_boxes(truths=True)
        plt.imshow(image)
        plt.show(block=True)

    else:
        print('Loading the testing dataset...')
        transform = tv.Compose([tv.ToTensor()])
        testing_dataset = dataset.AerialImagesDataset(csv_test_path, img_test_path, dim, classes, transform=transform)
        print('Dataset ready')

        print('Loading the training dataloader...')
        test_loader = DataLoader(dataset=testing_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('Training dataloader ready')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i, (image, annotations) in enumerate(test_loader):
            image = image.to(device)
            annotations = annotations.reshape(-1, 49 * 9)
            with torch.no_grad():
                outputs = network(image)

            #img = plt.imread(*img_path)
            final_pred = utils.FinalPredictions(outputs.cpu().to(torch.float32), annotations.to(torch.float32))
            # print(f'Detection stats: {utils.all_detections}')
            # print(f'No of positives: {utils.positives}')
            # if i == 2:
            #     sys.exit()
            # annt_test = utils.FinalPredictions(annotations, annotations)
            # plt.figure()
            # final_pred.draw_boxes()
            # annt_test.draw_boxes(other_color=True)
            # plt.imshow(img)
            # plt.show(block=True)
            # test = utils.all_detections

        ap1 = AveragePrecision(utils.all_detections['plane'], utils.positives['plane'])
        ap2 = AveragePrecision(utils.all_detections['ship'], utils.positives['ship'])
        ap3 = AveragePrecision(utils.all_detections['tennis-court'], utils.positives['tennis-court'])
        ap4 = AveragePrecision(utils.all_detections['swimming-pool'], utils.positives['swimming-pool'])

        print(ap1.get_average_precision())
        print(ap2.get_average_precision())
        print(ap3.get_average_precision())
        print(ap4.get_average_precision())

        mAP = (ap1.get_average_precision() + ap2.get_average_precision() + ap3.get_average_precision() + ap4.get_average_precision()) / 4
        print(f'mAP: {mAP}')








