import sys

import yaml
import customDataset as dataset
import model
import numpy as np
import time
import torchvision.transforms as tv
import torch.utils.data
from torch.utils.data import DataLoader
import utils


def loss_calc(outputs, truth):
    niu_coord = float(5)
    niu_noobj = float(0.5)

    loss = torch.tensor(0, dtype=torch.float32)
    for i in range(outputs.shape[0]):
        one_img_loss = 0
        # loss for bbox coords
        for j in range(0, outputs.shape[1], 42):
            for k in range(2):
                if truth[i, j + (k*21)] == 1:
                    x_out = outputs[i, j + (k*21) + 1]
                    x_truth = truth[i, j + (k*21) + 1]

                    y_out = outputs[i, j + (k * 21) + 2]
                    y_truth = truth[i, j + (k * 21) + 2]

                    w_out = outputs[i, j + (k * 21) + 3]
                    w_truth = truth[i, j + (k * 21) + 3]

                    h_out = outputs[i, j + (k * 21) + 4]
                    h_truth = truth[i, j + (k * 21) + 4]

                    if not w_out >= 0:
                        w_out = 0
                    if not h_out >= 0:
                        h_out = 0

                    one_img_loss += (x_out-x_truth)**2 + (y_out - y_truth)**2 + \
                                    (w_out - w_truth)**2 + (h_out - h_truth) ** 2

        one_img_loss *= niu_coord

        # loss if object is in cell
        for j in range(0, outputs.shape[1], 42):
            for k in range(2):
                if truth[i, j + (k*21)] == 1:
                    bbox_out = (outputs[i, j+(k*21)+1], outputs[i, j+(k*21)+2], outputs[i, j+(k*21)+3], outputs[i, j+(k*21)+4])
                    bbox_truth = (truth[i, j+(k*21)+1], truth[i, j+(k*21)+2], truth[i, j+(k*21)+3], truth[i, j+(k*21)+4])
                    c_truth = utils.get_iou(bbox_out, bbox_truth)
                    c_out = outputs[i, j + (k * 21)]
                    if c_truth < 0 or c_truth > 1:
                        one_img_loss += 0
                    else:
                        one_img_loss += (c_out-c_truth)**2

        # loss if no object is in the cell
        for j in range(0, outputs.shape[1], 42):
            for k in range(2):
                if truth[i, j + (k*21)] == 0:
                    c_out = outputs[i, j + (k*21)]
                    c_truth = float(0)
                    one_img_loss += (c_out-c_truth)**2
        one_img_loss *= niu_noobj

        # loss for class probabilities
        for j in range(0, outputs.shape[1], 42):
            for k in range(2):
                if truth[i, j + (k*21)] == 1:
                    for p in range(16):
                        p_out = outputs[i, j + (k*21) + 5 + p]
                        p_truth = truth[i, j + (k*21) + 5 + p]
                        one_img_loss += (p_out-p_truth)**2

        loss += one_img_loss
    return loss/outputs.shape[0]


if __name__ == "__main__":
    with open('configs/dataset-config.yml') as f:
        dataset_paths = yaml.safe_load(f)

    root_csv = dataset_paths['train_labels_csv']
    root_img = dataset_paths['train_images_path']
    dim = dataset_paths['img_dim']
    classes = dataset_paths['no_of_classes']

    print('Loading the dataset...')
    aerial_dataset = dataset.AerialImagesDataset(root_csv, root_img, dim, classes, transform=tv.ToTensor())
    print('Dataset ready')

    network = model.NetworkModel()

    print('Loading the dataloader...')
    dataloader = DataLoader(dataset=aerial_dataset, batch_size=64, shuffle=True, num_workers=1)
    print('Dataloader ready')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.0001
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

    print('Begin training loop')

    running_loss = 0
    EPOCHS = 10
    batch_no = 0
    torch.autograd.set_detect_anomaly(True)
    network.train()

    # init plotting objects
    loss_plot = utils.DynamicUpdate('Loss per batch')
    avg_time_plot = utils.DynamicUpdate('Average processing time')
    #epoch_loss_plot = utils.DynamicUpdate('Loss per epoch')

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}')
        for i, (image, annotations) in enumerate(dataloader):
            loop_begin = time.time_ns()
            image = image.to(device)
            annotations = annotations.reshape(-1, 49*42).to(device)

            outputs = network(image)
            assert annotations.shape == outputs.shape
            loss = loss_calc(outputs, annotations)
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            running_loss += loss.item()

            # evaluate loss after more batches
            if i % 1 == 0:
                loop_end = time.time_ns()
                duration = (loop_end - loop_begin) * (10 ** (-9))
                last_loss = running_loss / 1  # loss per batch
                print(f'Completed batch {batch_no+1} in {duration} seconds, loss: {last_loss}')
                utils.plot_dynamic_graph(loss_plot, last_loss, batch_no+1)
                utils.plot_dynamic_graph(avg_time_plot, duration, batch_no+1)
                batch_no += 1
                running_loss = 0
        # TODO Add epoch loss
