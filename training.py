import sys

import yaml
import customDataset as dataset
import model
import time
import torchvision.transforms as tv
import torch.utils.data
from torch.utils.data import DataLoader
import utils
from metrics import AveragePrecision


def loss_calc(outputs, truth):
    niu_coord = float(5)
    niu_noobj = float(0.5)

    loss = torch.tensor(0, dtype=torch.float32)
    if torch.cuda.is_available():
        loss = torch.tensor(0, device=torch.device('cuda'), dtype=torch.float32)
    for i in range(outputs.shape[0]):
        one_img_loss = torch.tensor(0, dtype=torch.float32)
        if torch.cuda.is_available():
            one_img_loss = torch.tensor(0, device=torch.device('cuda'), dtype=torch.float32)
        # loss for bbox coords
        for j in range(0, outputs.shape[1], 5 + no_of_classes):
            for k in range(objects_in_grid):
                if truth[i, j + (k * (5 + no_of_classes))] == 1:
                    x_out = outputs[i, j + (k * (5 + no_of_classes)) + 1]
                    x_truth = truth[i, j + (k * (5 + no_of_classes)) + 1]

                    y_out = outputs[i, j + (k * (5 + no_of_classes)) + 2]
                    y_truth = truth[i, j + (k * (5 + no_of_classes)) + 2]

                    w_out = outputs[i, j + (k * (5 + no_of_classes)) + 3]
                    w_truth = truth[i, j + (k * (5 + no_of_classes)) + 3]

                    h_out = outputs[i, j + (k * (5 + no_of_classes)) + 4]
                    h_truth = truth[i, j + (k * (5 + no_of_classes)) + 4]

                    if not w_out > 0:
                        w_out = torch.tensor(0.1)
                    if not h_out > 0:
                        h_out = torch.tensor(0.1)

                    one_img_loss += (x_out - x_truth) ** 2 + (y_out - y_truth) ** 2 + \
                                    (torch.sqrt(w_out) - torch.sqrt(w_truth)) ** 2 + \
                                    (torch.sqrt(h_out) - torch.sqrt(h_truth)) ** 2

        one_img_loss *= niu_coord

        # iou loss if object is in cell
        for j in range(0, outputs.shape[1], 5 + no_of_classes):
            for k in range(objects_in_grid):
                if truth[i, j + (k * (5 + no_of_classes))] == 1:
                    bbox_out = (outputs[i, j + (k * (5 + no_of_classes)) + 1],
                                outputs[i, j + (k * (5 + no_of_classes)) + 2],
                                outputs[i, j + (k * (5 + no_of_classes)) + 3],
                                outputs[i, j + (k * (5 + no_of_classes)) + 4])

                    bbox_truth = (truth[i, j + (k * (5 + no_of_classes)) + 1],
                                  truth[i, j + (k * (5 + no_of_classes)) + 2],
                                  truth[i, j + (k * (5 + no_of_classes)) + 3],
                                  truth[i, j + (k * (5 + no_of_classes)) + 4])
                    iou_pred_truth = utils.get_iou_new(bbox_out, bbox_truth)
                    iou_desired = 1
                    if iou_pred_truth < 0 or iou_pred_truth > 1:
                        one_img_loss += 0
                    else:
                        one_img_loss += (iou_desired - iou_pred_truth) ** 2

        # loss for objectness
        for j in range(0, outputs.shape[1], 5 + no_of_classes):
            for k in range(objects_in_grid):
                if truth[i, j + (k * (5 + no_of_classes))] == 1:
                    c_desired = 1
                    c_out = outputs[i, j + (k * (5 + no_of_classes))]
                    one_img_loss += (c_desired - c_out) ** 2

        # loss if no object is in the cell
        noobj_loss = 0
        for j in range(0, outputs.shape[1], 5 + no_of_classes):
            for k in range(objects_in_grid):
                if truth[i, j + (k * (5 + no_of_classes))] == 0:
                    c_out = outputs[i, j + (k * (5 + no_of_classes))]
                    c_truth = float(0)
                    noobj_loss += (c_out - c_truth) ** 2
        one_img_loss += niu_noobj * noobj_loss

        # loss for class probabilities
        for j in range(0, outputs.shape[1], 5 + no_of_classes):
            for k in range(objects_in_grid):
                if truth[i, j + (k * (5 + no_of_classes))] == 1:
                    for p in range(no_of_classes):
                        p_out = outputs[i, j + (k * (5 + no_of_classes)) + 5 + p]
                        p_truth = truth[i, j + (k * (5 + no_of_classes)) + 5 + p]
                        one_img_loss += (p_out - p_truth) ** 2

        loss += one_img_loss
    return loss / outputs.shape[0]


def validation_loop(validation_loader, network):
    network.eval()
    network.set_testing()
    with torch.no_grad():
        # running_vloss = 0
        c = '|'
        sys.stdout.write('Computing avarage precision for validation set...\n')
        for k, (img, annt) in enumerate(validation_loader):
            img = img.to(device)
            annt = annt.reshape(-1, 49 * (5 + no_of_classes))

            out = network(img)
            for j in range(annt.shape[0]):
                _ = utils.FinalPredictions(out[j].to(torch.float32).cpu(), annt[j].to(torch.float32))
            # sys.stdout.write(c)
            # c += '|'
            # sys.stdout.flush()

        #    val_loss = loss_calc(out, annt)
        #     print(f'Validation {k} loss: {val_loss}')
        #     running_vloss += val_loss.item()
        # print(f'Validation loss: {running_vloss/k}')
        # TODO mAP computing should be configurable and done in a separate function, maybe in utils
        # TODO Add loading animation (minimal priority)
        ap_planes = AveragePrecision(utils.all_detections['plane'], utils.positives['plane'])
        ap_ship = AveragePrecision(utils.all_detections['ship'], utils.positives['ship'])
        ap_tennis = AveragePrecision(utils.all_detections['tennis-court'], utils.positives['tennis-court'])
        ap_swimming = AveragePrecision(utils.all_detections['swimming-pool'], utils.positives['swimming-pool'])
        mAP = (ap_planes.get_average_precision() + ap_ship.get_average_precision() + ap_tennis.get_average_precision() + ap_swimming.get_average_precision()) / 4
        print(f'Validation mAP: {mAP}')
        for key in utils.positives.keys():
            utils.positives[key] = 0
        for key in utils.all_detections.keys():
            utils.all_detections[key] = []
        network.set_training()
        print('Exit or will continue in 10s...')
        time.sleep(10)
        return mAP


objects_in_grid = 0
no_of_classes = 0
train_csv = ''
train_img = ''
validation_csv = ''
validation_img = ''
dim = 0
classes = 0
last_state_file = ''
best_state_file = ''
checkpoint = False
batch_size = 0
epochs = 0
learning_rate = 0


def init():
    with open('configs/model-config.yaml') as f:
        training_cfg = yaml.safe_load(f)

    global objects_in_grid
    objects_in_grid = training_cfg['general']['objects_in_grid']
    global no_of_classes
    no_of_classes = training_cfg['general']['no_of_classes']
    global train_csv
    train_csv = training_cfg['training']['train_labels_csv']
    global train_img
    train_img = training_cfg['training']['train_images_path']
    global validation_csv
    validation_csv = training_cfg['training']['validation_csv_path']
    global validation_img
    validation_img = training_cfg['training']['validation_images_path']
    global dim
    dim = training_cfg['general']['img_dim']
    global classes
    classes = training_cfg['general']['no_of_classes']
    global last_state_file
    last_state_file = training_cfg['training']['last_state_file']
    global best_state_file
    best_state_file = training_cfg['training']['best_state_file']
    global checkpoint
    checkpoint = training_cfg['training']['checkpoint']
    global batch_size
    batch_size = training_cfg['training']['batch_size']
    global epochs
    epochs = training_cfg['training']['epochs']
    global learning_rate
    learning_rate = training_cfg['training']['learning_rate']


if __name__ == "__main__":
    init()

    print('Loading the training dataset...')
    transform = tv.Compose([tv.ToTensor()])
    training_dataset = dataset.AerialImagesDataset(
        train_csv, train_img, dim, classes, objects_in_grid, transform=transform)
    print('Dataset ready')

    print('Loading the validation dataset...')
    validation_dataset = dataset.AerialImagesDataset(
        validation_csv, validation_img, dim, classes, objects_in_grid, transform=tv.ToTensor())
    print('Dataset ready')

    print('Loading the training dataloader...')
    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Training dataloader ready')

    print('Loading the validation dataloader...')
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Validation dataloader ready')

    network = model.NetworkModel(no_of_classes=no_of_classes, obj_in_grid=objects_in_grid)
    if torch.cuda.is_available():
        network.cuda()

    if checkpoint:
        print('Reload from checkpoint')
        try:
            network.load_state_dict(torch.load(best_state_file))
        except RuntimeError:
            print('Weights from file don\'t match model\'s weights shape, please check number of classes'
                  ' and number of objects to be detected in a grid cell')
            print('exiting...')
            time.sleep(2)
            sys.exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

    print('Begin training loop')

    # init loss tracking variables
    running_loss = 0
    batch_no = 0
    group_batch_loss = 0
    torch.autograd.set_detect_anomaly(True)
    network.train()

    losses = []
    proc_times = []

    loop_begin = 0
    max_ap = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        epoch_begin = time.time_ns()
        for i, (image, annotations) in enumerate(train_loader):
            if i == 0:
                loop_begin = time.time_ns()
            image = image.to(device)
            annotations = annotations.reshape(-1, 49 * (5 + no_of_classes)).to(device)

            outputs = network(image)
            assert annotations.shape == outputs.shape
            loss = loss_calc(outputs.to(torch.float32), annotations.to(torch.float32))
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            running_loss += loss.item()
            group_batch_loss += loss.item()

            # loss per 10 batches
            if i % 10 == 0 and i != 0:
                # reporting after 10 batches
                loop_end = time.time_ns()
                duration = (loop_end - loop_begin) * (10 ** (-9))
                loop_begin = time.time_ns()
                group_batch_loss = group_batch_loss / 10
                print(f'Completed 10 batches at {batch_no + 1} in {duration} seconds, loss: {group_batch_loss}')
                group_batch_loss = 0
            batch_no += 1

        # reporting after 1 epoch
        epoch_loss = running_loss / batch_no
        epoch_end = time.time_ns()
        epoch_duration = (epoch_end - epoch_begin) * (10 ** 9)
        print(f'Epoch {epoch + 1} ended in {epoch_duration}, loss: {epoch_loss}')
        losses.append(epoch_loss)
        proc_times.append(epoch_duration)
        ap = validation_loop(validation_loader, network)
        if ap > max_ap:
            print(f'Saving best version of weights, average precision = {ap}')
            max_ap = ap
            torch.save(network.state_dict(), best_state_file)

        print('Saving last version of weights...')
        torch.save(network.state_dict(), last_state_file)
        network.train()
        running_loss = 0
        batch_no = 0
