import torch
import yaml
import customDataset as dataset
import torchvision.transforms as tv
from torch.utils.data import DataLoader
import utils
from metrics import AveragePrecision
import matplotlib.pyplot as plt
from model import NetworkModel

if __name__ == "__main__":
    with open('configs/testing-config.yaml') as f:
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

    print('Loading the testing dataset...')
    transform = tv.Compose([tv.ToTensor()])
    testing_dataset = dataset.AerialImagesDataset(csv_test_path, img_test_path, dim, classes, transform=transform)
    print('Dataset ready')

    print('Loading the training dataloader...')
    test_loader = DataLoader(dataset=testing_dataset, batch_size=1, shuffle=True, num_workers=1)
    print('Training dataloader ready')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO Here just pass the output and log the metrics at the end
    for i, (image, annotations, img_path) in enumerate(test_loader):
        image = image.to(device)
        annotations = annotations.reshape(-1, 49 * 6)
        with torch.no_grad():
            outputs = network(image)

        img = plt.imread(*img_path)
        final_pred = utils.FinalPredictions(outputs.cpu(), annotations)
        # annt_test = utils.FinalPredictions(annotations, annotations)
        # plt.figure()
        # final_pred.draw_boxes()
        # annt_test.draw_boxes(other_color=True)
        # plt.imshow(img)
        # plt.show(block=True)
        # test = utils.all_detections

    ap = AveragePrecision(utils.all_detections, utils.positives)
    print(ap.get_average_precision())








