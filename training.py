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
        for j in range(0, outputs.shape[1], 9):
            for k in range(1):
                if truth[i, j + (k*21)] == 1:
                    x_out = outputs[i, j + (k*21) + 1]
                    x_truth = truth[i, j + (k*21) + 1]

                    y_out = outputs[i, j + (k * 21) + 2]
                    y_truth = truth[i, j + (k * 21) + 2]

                    w_out = outputs[i, j + (k * 21) + 3]
                    w_truth = truth[i, j + (k * 21) + 3]

                    h_out = outputs[i, j + (k * 21) + 4]
                    h_truth = truth[i, j + (k * 21) + 4]

                    if not w_out > 0:
                        w_out = torch.tensor(0.1)
                    if not h_out > 0:
                        h_out = torch.tensor(0.1)

                    one_img_loss += (x_out-x_truth)**2 + (y_out - y_truth)**2 + \
                                    (torch.sqrt(w_out) - torch.sqrt(w_truth))**2 + \
                                    (torch.sqrt(h_out) - torch.sqrt(h_truth)) ** 2

        one_img_loss *= niu_coord

        # iou loss if object is in cell
        for j in range(0, outputs.shape[1], 9):
            for k in range(1):
                if truth[i, j + (k*21)] == 1:
                    bbox_out = (outputs[i, j+(k*21)+1], outputs[i, j+(k*21)+2], outputs[i, j+(k*21)+3], outputs[i, j+(k*21)+4])
                    bbox_truth = (truth[i, j+(k*21)+1], truth[i, j+(k*21)+2], truth[i, j+(k*21)+3], truth[i, j+(k*21)+4])
                    iou_pred_truth = utils.get_iou_new(bbox_out, bbox_truth)
                    iou_desired = 1
                    if iou_pred_truth < 0 or iou_pred_truth > 1:
                        one_img_loss += 0
                    else:
                        one_img_loss += (iou_desired-iou_pred_truth)**2

        # loss for objectness
        for j in range(0, outputs.shape[1], 9):
            for k in range(1):
                if truth[i, j + (k*21)] == 1:
                    c_desired = 1
                    c_out = outputs[i, j + (k * 21)]
                    one_img_loss += (c_desired-c_out)**2

        # loss if no object is in the cell
        noobj_loss = 0
        for j in range(0, outputs.shape[1], 9):
            for k in range(1):
                if truth[i, j + (k*21)] == 0:
                    c_out = outputs[i, j + (k*21)]
                    c_truth = float(0)
                    noobj_loss += (c_out-c_truth)**2
        one_img_loss += niu_noobj*noobj_loss

        # loss for class probabilities
        for j in range(0, outputs.shape[1], 9):
            for k in range(1):
                if truth[i, j + (k*21)] == 1:
                    for p in range(4):
                        p_out = outputs[i, j + (k*21) + 5 + p]
                        p_truth = truth[i, j + (k*21) + 5 + p]
                        one_img_loss += (p_out-p_truth)**2

        loss += one_img_loss
    return loss/outputs.shape[0]


def validation_loop(validation_loader, network):
    network.eval()
    network.set_testing()
    with torch.no_grad():
        # running_vloss = 0
        c = '|'
        sys.stdout.write('Computing avarage precision for validation set...\n')
        for k, (img, annt) in enumerate(validation_loader):
            img = img.to(device)
            annt = annt.reshape(-1, 49 * 6)

            out = network(img)
            for j in range(annt.shape[0]):
                _ = utils.FinalPredictions(out[j].to(torch.float32).cpu(), annt[j].to(torch.float32))
            sys.stdout.write(c)
            c += '|'
            sys.stdout.flush()

        #    val_loss = loss_calc(out, annt)
        #     print(f'Validation {k} loss: {val_loss}')
        #     running_vloss += val_loss.item()
        # print(f'Validation loss: {running_vloss/k}')
        ap_planes = AveragePrecision(utils.all_detections['plane'], utils.positives['plane'])
        ap_ship = AveragePrecision(utils.all_detections['ship'], utils.positives['ship'])
        ap_tennis = AveragePrecision(utils.all_detections['tennis-court'], utils.positives['tennis-court'])
        ap_swimming = AveragePrecision(utils.all_detections['swimming-pool'], utils.positives['swimming-pool'])
        mAP = (ap_planes.get_average_precision() + ap_ship.get_average_precision() + ap_tennis.get_average_precision() + ap_swimming.get_average_precision()) / 4
        print(f'Validation mAP: {mAP}')
        utils.positives = 0
        utils.all_detections = []
        network.set_training()
        print('Exit or will continue in 10s...')
        time.sleep(10)
        return mAP


if __name__ == "__main__":
    with open('configs/dataset-config.yml') as f:
        dataset_paths = yaml.safe_load(f)

    train_csv = dataset_paths['train_labels_csv']
    train_img = dataset_paths['train_images_path']
    validation_csv = dataset_paths['validation_csv_path']
    validation_img = dataset_paths['validation_images_path']
    dim = dataset_paths['img_dim']
    classes = dataset_paths['no_of_classes']
    last_state_file = dataset_paths['last_state_file']
    best_state_file = dataset_paths['best_state_file']
    checkpoint = dataset_paths['checkpoint']

    print('Loading the training dataset...')
    transform = tv.Compose([tv.ToTensor()])
    training_dataset = dataset.AerialImagesDataset(train_csv, train_img, dim, classes, transform=transform)
    print('Dataset ready')

    print('Loading the validation dataset...')
    validation_dataset = dataset.AerialImagesDataset(validation_csv, validation_img, dim, classes, transform=tv.ToTensor())
    print('Dataset ready')

    print('Loading the training dataloader...')
    train_loader = DataLoader(dataset=training_dataset, batch_size=16, shuffle=True, num_workers=1)
    print('Training dataloader ready')

    print('Loading the validation dataloader...')
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=16, shuffle=True, num_workers=1)
    print('Validation dataloader ready')

    network = model.NetworkModel()
    if torch.cuda.is_available():
        network.cuda()

    if checkpoint:
        print('Reload from checkpoint')
        network.load_state_dict(torch.load(last_state_file))

    # for param in network.parameters():
    #     if len(param.data.shape) == 4:
    #         param.requires_grad = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.0001
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

    print('Begin training loop')

    running_loss = 0
    EPOCHS = 100
    batch_no = 0
    group_batch_loss = 0
    torch.autograd.set_detect_anomaly(True)
    network.train()

    # init plotting objects
    loss_plot = utils.DynamicUpdate('Loss per batch')
    avg_time_plot = utils.DynamicUpdate('Average processing time')
    epoch_loss_plot = utils.DynamicUpdate('Loss per epoch', max_x=EPOCHS)
    loop_begin = 0
    max_ap = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}')
        for i, (image, annotations) in enumerate(train_loader):
            if i == 0:
                loop_begin = time.time_ns()
            image = image.to(device)
            annotations = annotations.reshape(-1, 49*6).to(device)

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
                group_batch_loss = group_batch_loss/10
                print(f'Completed 10 batches at {batch_no+1} in {duration} seconds, loss: {group_batch_loss}')
            #utils.plot_dynamic_graph(loss_plot, loss.item(), batch_no+1)
            #utils.plot_dynamic_graph(avg_time_plot, duration, batch_no+1)
            batch_no += 1

        # reporting after 1 epoch
        epoch_loss = running_loss/batch_no
        print(f'Epoch {epoch+1} loss: {epoch_loss}')
        #utils.plot_dynamic_graph(epoch_loss_plot, epoch_loss, epoch)
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
