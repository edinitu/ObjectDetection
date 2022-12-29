import yaml
import customDataset as dataset
import model
import numpy as np
import time
import torchvision.transforms as tv
import torch.utils.data
from torch.utils.data import DataLoader


def loss_calc(outputs, truth):
    niu_coord = float(5)
    niu_noobj = float(0.5)

    loss = torch.tensor(0, dtype=torch.float32)
    for i in range(outputs.shape[0]):
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

                    loss += (x_out-x_truth)**2 + (y_out - y_truth)**2 + \
                            (torch.sqrt(abs(w_out)) - torch.sqrt(abs(w_truth)))**2 + \
                            (torch.sqrt(abs(h_out)) - torch.sqrt(abs(h_truth))) ** 2
                    loss *= niu_coord

        for j in range(0, outputs.shape[1], 42):
            for k in range(2):
                if truth[i, j + (k*21)] == 1:
                    c_out = outputs[i, j + (k*21)]
                    c_truth = float(1)
                    loss += (c_out-c_truth)**2

        for j in range(0, outputs.shape[1], 42):
            for k in range(2):
                if truth[i, j + (k*21)] == 0:
                    c_out = outputs[i, j + (k*21)]
                    c_truth = float(0)
                    loss += (c_out-c_truth)**2
                    loss *= niu_noobj

        for j in range(0, outputs.shape[1], 42):
            for k in range(2):
                if truth[i, j + (k*21)] == 1:
                    for p in range(16):
                        p_out = outputs[i, j + (k*21) + 5 + p]
                        p_truth = truth[i, j + (k*21) + 5 + p]
                        loss += (p_out-p_truth)**2

    return loss


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
    dataloader = DataLoader(dataset=aerial_dataset, batch_size=4, shuffle=True, num_workers=1)
    print('Dataloader ready')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    print('Begin training loop')
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
        loop_end = time.time_ns()
        duration = (loop_end - loop_begin) * (10**(-9))
        print(f'Completed loop {i} in {duration} seconds, loss: {loss.item()}')






