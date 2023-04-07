import os.path
import sys
import threading
import time
import numpy as np
import torch
import yaml
import customDataset as dataset
import torchvision.transforms as tv
from torch.utils.data import DataLoader
import utils
import pandas as pd
import matplotlib.pyplot as plt
from model import NetworkModel

img_test_path: str
csv_test_path: str
dim: int
no_of_classes: int
weights: dict
obj_in_grid: int
one_dataset_image: bool
one_random_image: bool
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
    global one_dataset_image
    one_dataset_image = testing_cfg['oneDatasetImage']
    global one_random_image
    one_random_image = testing_cfg['oneRandomImage']
    if one_dataset_image or one_random_image:
        global image_path
        image_path = testing_cfg['image']
        global draw_ground_truth
        draw_ground_truth = testing_cfg['draw_ground_truth']

    with open('configs/pre-processing-config.yaml') as f:
        preproc_config = yaml.safe_load(f)

    global labels
    labels = preproc_config['processImages']['labels']


def test_img_section(key, d):
    d[key] = np.float16(d[key])
    d[key] = utils.image_checks(d[key], dim, dim)
    d[key], annotations = utils.torch_prepare(d[key], np.zeros((1, 5)))

    with torch.no_grad():
        outputs = network(cropped[key])

    final_pred = utils.FinalPredictions(outputs.cpu(), annotations)
    full_scale_pred.add_prediction(key, final_pred)


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

    if one_dataset_image:
        print('Testing one image from DOTA dataset')
        annotations = pd.read_csv(os.path.join(csv_test_path, image_path + '.csv'), header=None)
        annotations = np.array(annotations)

        image = plt.imread(os.path.join(img_test_path, image_path + '.png'))
        image = image.astype(np.float16)
        image = utils.image_checks(image, dim, dim)
        image, annotations = utils.torch_prepare(image, annotations)

        with torch.no_grad():
            start = time.time_ns()
            outputs = network(image)
            end = time.time_ns()

        inference_time = (end - start) * (10**(-6))
        print(f'Inference time: {inference_time}')
        print('Detected objects: ')
        plt.figure()
        final_pred = utils.FinalPredictions(outputs.cpu(), annotations)
        annt_test = utils.FinalPredictions(annotations, annotations)
        image = plt.imread(os.path.join(img_test_path, image_path + '.png'))
        final_pred.draw_boxes()
        if draw_ground_truth:
            annt_test.draw_boxes(truths=True)
        plt.imshow(image)
        plt.show(block=True)

    elif one_random_image:
        print('Testing one random image')
        cropped = utils.crop_img(image_path, dim)
        full_scale_pred = utils.FullScalePrediction()
        threads = []
        for key in cropped:
            thread = threading.Thread(test_img_section(key, cropped))
            threads.append(thread)

        [t.start() for t in threads]
        [t.join() for t in threads]

        plt.figure()
        img = plt.imread(image_path)
        full_scale_pred.to_full_scale()
        full_scale_pred.draw()
        plt.imshow(img)
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

        mAP = utils.get_mAP()
        print(f'mAP: {mAP}')

        avg_inf_time = np.sum(np.asarray(durations)) / len(durations)
        print(f'Average inference time: {avg_inf_time} ms')

        true_pos_count = utils.get_TP_count()
        all_pos_count = utils.get_P_count()
        print(f'Objects detected correctly count: {true_pos_count}'
              f'\n All objects in testing set: {all_pos_count} \n'
              f'Ratio: {true_pos_count/all_pos_count} \n'
              f'Average ratio per image: {utils.get_avg_ratio()}')
