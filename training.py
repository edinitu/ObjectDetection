import yaml
import customDataset as dataset
import model
import numpy as np
import time
import torchvision.transforms as tv
import torch.utils.data
from torch.utils.data import DataLoader


def loss_calc(outputs, truth):
    # TODO implement loss calculation
    return torch.ones(1, requires_grad=True)


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
    learning_rate = 0.01
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    print('Begin training loop')
    for i, (image, annotations) in enumerate(dataloader):
        image = image.to(device)
        annotations = annotations.reshape(-1, 49*42).to(device)

        outputs = network(image)
        assert annotations.shape == outputs.shape
        loss = loss_calc(outputs, annotations)
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()






