import yaml
import customDataset as dataset
import model
import numpy as np
import time
import torchvision.transforms as tv
import torch.utils.data
from torch.utils.data import DataLoader


def my_collate_fn(data):
    img_list = []
    annt_list = []
    for image, annt in data:
        assert image.shape == (448, 448, 3)
        assert annt.shape == (49, 42, 1)
        img_list.append(image)
        annt_list.append(annt)
    return torch.tensor(np.asarray(img_list)), torch.tensor(np.asarray(annt_list))


if __name__ == "__main__":
    with open('configs/dataset-config.yml') as f:
        dataset_paths = yaml.safe_load(f)

    root_csv = dataset_paths['train_labels_csv']
    root_img = dataset_paths['train_images_path']
    dim = dataset_paths['img_dim']
    classes = dataset_paths['no_of_classes']

    aerial_dataset = dataset.AerialImagesDataset(root_csv, root_img, dim, classes)
    network = model.NetworkModel()
    dataloader = DataLoader(dataset=aerial_dataset, batch_size=64, shuffle=True, num_workers=1, collate_fn=my_collate_fn)
    dataiter = iter(dataloader)
    data = dataiter.__next__()
    image, annotations = data
    print(type(image), image.shape, type(annotations), annotations.shape)
