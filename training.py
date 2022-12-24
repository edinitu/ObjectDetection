import yaml
import customDataset as dataset
import model
import torchvision.transforms as tv
import torch.utils.data
from torch.utils.data import DataLoader

if __name__ == "__main__":
    with open('configs/dataset-config.yml') as f:
        dataset_paths = yaml.safe_load(f)

    root_csv = dataset_paths['train_labels_csv']
    root_img = dataset_paths['train_images_path']
    dim = dataset_paths['img_dim']
    classes = dataset_paths['no_of_classes']

    aerial_dataset = dataset.AerialImagesDataset(root_csv, root_img, dim, classes)#, transform=tv.Compose([
              #tv.ToTensor()
            #]))
    network = model.NetworkModel()

    dataloader = DataLoader(dataset=aerial_dataset, batch_size=4, shuffle=True, num_workers=1)
    dataiter = iter(dataloader)
    data = dataiter.__next__()
    image, annotations = data
    print(type(image), image.shape, type(annotations), annotations.shape)
