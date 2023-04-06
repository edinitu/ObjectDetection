import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import torchvision.transforms as tv
import torch
import yaml
import customDataset as dataset
from showImageFromDataset import ImageElement
from metrics import PredictionStats, TRUE_POSITIVE, FALSE_POSITIVE, AveragePrecision


class Plotting:
    """
    Class for plotting training statistics: loss per batch, average processing time per batch etc.
    """
    def __init__(self, epochs, loss_list, mAP_list, objects_detected, proc_times, path):
        self.epochs = epochs
        self.loss_list = loss_list
        self.mAP_list = mAP_list
        self.objects_detected = objects_detected
        self.proc_times = proc_times
        self.path = path

        self.plot_and_save()

    def plot_and_save(self):
        epoch_list = [i for i in range(self.epochs)]
        plt.plot(epoch_list, self.loss_list)
        plt.savefig(os.path.join(self.path, 'training_loss.png'))
        plt.clf()
        plt.plot(epoch_list, self.mAP_list)
        plt.savefig(os.path.join(self.path, 'validation_mAP.png'))
        plt.clf()
        plt.plot(epoch_list, self.proc_times)
        plt.savefig(os.path.join(self.path, 'epoch_processing_time_minutes.png'))
        plt.clf()
        plt.plot(epoch_list, self.objects_detected)
        plt.savefig(os.path.join(self.path, 'objects_detected_count.png'))
        plt.close()


labels: dict
classes_dict: dict
all_detections: dict
positives: dict
no_of_classes: int
iou_conf_threshold: int
iou_nms_threshold: int
iou_TP_threshold: int
ratios: list
dim: int


def init():
    with open('configs/pre-processing-config.yaml') as f:
        preproc_config = yaml.safe_load(f)

    global labels
    labels = preproc_config['processImages']['labels']

    global classes_dict
    classes_dict = {v: k for k, v in labels.items()}
    global all_detections
    all_detections = {k: [] for k in labels}
    global positives
    positives = {k: 0 for k in labels}
    global no_of_classes
    no_of_classes = len(labels.keys())
    global dim
    dim = preproc_config['processImages']['cropped_img_dim']

    with open('configs/model-config.yaml') as f:
        model_config = yaml.safe_load(f)

    general_cfg = model_config['general']
    global iou_conf_threshold
    iou_conf_threshold = general_cfg['iou_conf_threshold']
    global iou_nms_threshold
    iou_nms_threshold = general_cfg['iou_nms_threshold']
    global iou_TP_threshold
    iou_TP_threshold = general_cfg['iou_TP_threshold']

    global ratios
    ratios = []


def get_label(classes_list):
    max_idx = 0
    for i in range(1, len(classes_list), 1):
        if classes_list[i] > classes_list[max_idx]:
            max_idx = i

    return classes_dict[max_idx]


def get_mAP():
    count = 0
    sum = 0
    for key in labels.keys():
        ap = AveragePrecision(all_detections[key], positives[key])
        sum += ap.get_average_precision()
        count += 1

    return sum / count


def get_TP_count():
    count = 0
    for item in all_detections.values():
        for elem in item:
            if elem.get_confusion() == 'TP':
                count += 1

    return count


def get_P_count():
    count = 0
    for elem in positives.values():
        count += elem

    return count


def get_avg_ratio():
    return float(np.sum(np.asarray(ratios)) / len(ratios))


class FullScalePrediction:
    """
        Class used for displaying the predicted bounding box over an image with random dimension.
        Holds the list of predictions from all the fixed dimension pieces that were sent as input
        in the network.
    """
    def __init__(self):
        self.final_predictions_list = {}

    def add_prediction(self, tup, prediction):
        self.final_predictions_list[tup] = prediction

    def to_full_scale(self):
        # TODO This code works but is ugly. Prettify and also comment
        for key in self.final_predictions_list.keys():
            for class_key in self.final_predictions_list[key].get_grids().keys():
                for grid_id in self.final_predictions_list[key].get_grids()[class_key]:
                    new_bbox = []
                    t = 0
                    for point in self.final_predictions_list[key].get_grids()[class_key][grid_id].get_bounding_box():
                        if t % 2 == 0:
                            new_bbox.append(point + key[1])
                        else:
                            new_bbox.append(point + key[0])
                        t += 1
                    self.final_predictions_list[key].set_grid(class_key, grid_id, new_bbox)

    def draw(self):
        for elem in self.final_predictions_list.values():
            elem.draw_boxes()


def crop_img(img, dim) -> dict:
    im = Image.open(img)
    cropped_list = {}
    for i in range(0, im.size[0], dim):
        for j in range(0, im.size[1], dim):
            cropped = (j, i, j+dim, i+dim)
            cropped_list[(i, j)] = im.crop(cropped)
    im.close()
    return cropped_list


class FinalPredictions:
    no_of_grids = 49

    # TODO Add comments
    def __init__(self, outputs, truths):
        self.grids = {k: {} for k in labels}
        self.truths = {k: {} for k in labels}
        grid_id = 0
        outputs = torch.reshape(outputs, (49, 5 + no_of_classes))
        truths = torch.reshape(truths, (49, 5 + no_of_classes))
        count = 0
        for elem in truths:
            if elem[0] == 1:
                img_elem = self.build_img_elem(elem)
                self.truths[img_elem.get_label()][count] = img_elem
            count += 1

        for elem in outputs:
            if elem[0] < iou_conf_threshold:
                grid_id += 1
                continue
            img_elem = self.build_img_elem(elem)
            self.grids[img_elem.get_label()][grid_id] = img_elem
            grid_id += 1

        self.non_max_suppression()
        self.convert_to_dota()
        self.add_to_stats_list()

    # TODO Define unit test for this
    def non_max_suppression(self):
        for class_key in self.grids.keys():
            conf_id_map = {}
            for secondary_key in self.grids[class_key].keys():
                conf_id_map[self.grids[class_key][secondary_key].get_confidence()] = secondary_key

            sorted_conf = []
            for key in conf_id_map.keys():
                sorted_conf.append(key)
            sorted_conf.sort(reverse=True)

            for conf in sorted_conf:
                if conf_id_map[conf] not in self.grids[class_key]:
                    continue
                self.remove_overlapped_boxes(conf_id_map[conf], class_key)

    def remove_overlapped_boxes(self, reference_key, class_key):
        for key in range(self.no_of_grids):
            if key != reference_key and key in self.grids[class_key]:
                iou = get_iou_new(
                    convert_to_yolo_full_scale(self.grids[class_key][reference_key].get_yolo_bbox(), reference_key),
                    convert_to_yolo_full_scale(self.grids[class_key][key].get_yolo_bbox(), key)
                )
                if iou > iou_nms_threshold:
                    del self.grids[class_key][key]

    def add_to_stats_list(self):
        """
        Here we populate the list with all detections in the testing set. They can either be
        true positives or false positives. Also increment the number of ground truth positives.
        """
        tp_count = 0
        pos_count = 0
        for class_key in self.grids.keys():
            for pred_key in self.grids[class_key].keys():
                count = 0
                for truth_key in self.truths[class_key].keys():
                    iou = get_iou_new(
                        convert_to_yolo_full_scale(self.grids[class_key][pred_key].get_yolo_bbox(), pred_key),
                        convert_to_yolo_full_scale(self.truths[class_key][truth_key].get_yolo_bbox(), truth_key)
                    )
                    #    print(iou)
                    if iou > iou_TP_threshold:
                        all_detections[class_key].append(
                            PredictionStats(self.grids[class_key][pred_key].get_confidence(), TRUE_POSITIVE)
                        )
                        tp_count += 1
                        count += 1
                        break
                if count == 0:
                    all_detections[class_key].append(
                        PredictionStats(self.grids[class_key][pred_key].get_confidence(), FALSE_POSITIVE)
                    )

            global positives
            positives[class_key] += len(list(self.truths[class_key].keys()))
            pos_count += len(list(self.truths[class_key].keys()))

        try:
            ratios.append(float(tp_count/pos_count))
        except ZeroDivisionError:
            if tp_count == 0:
                ratios.append(float(1))
            else:
                ratios.append(float(0))

    def convert_to_dota(self):
        for class_key in self.grids.keys():
            for key in self.grids[class_key].keys():
                self.grids[class_key][key].convert_yolo_to_dota(key)

    def draw_boxes(self, truths=False):
        if truths:
            for class_key in self.grids.keys():
                for elem in self.grids[class_key].values():
                    elem.draw_box(color='red')
        else:
            for class_key in self.grids.keys():
                for elem in self.grids[class_key].values():
                    elem.draw_box()

    def build_img_elem(self, elem) -> ImageElement:
        img_elem = ImageElement()
        img_elem.set_yolo_bbox([elem[1], elem[2], elem[3], elem[4]])
        img_elem.set_confidence(elem[0])
        labels_to_get = []
        for i in range(len(labels)):
            labels_to_get.append(elem[5 + i])
        img_elem.set_label(get_label(labels_to_get))
        return img_elem

    def get_grids(self):
        return self.grids

    def set_grid(self, class_key, grid_id, value):
        self.grids[class_key][grid_id].set_bbox(value)


def grey2rgb(img):
    assert img.shape == (448, 448)
    new_img = np.zeros((448, 448, 3), dtype=np.float16)
    new_img[:, :, 0] = img
    new_img[:, :, 1] = img
    new_img[:, :, 2] = img
    return new_img


def get_iou_new(bbox1, bbox2):
    boxA = [bbox1[0] - bbox1[2] / 2, bbox1[1] - bbox1[3] / 2, bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2]
    boxB = [bbox2[0] - bbox2[2] / 2, bbox2[1] - bbox2[3] / 2, bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate the area of intersection rectangle
    interArea = max(torch.tensor(0), xB - xA + 1) * max(torch.tensor(0), yB - yA + 1)
    interArea = interArea.to(torch.float32)

    # Calculate the area of the two bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    boxAArea = boxAArea.to(torch.float32)
    boxBArea = boxBArea.to(torch.float32)

    # Calculate the IOU
    iou = interArea / (boxAArea + boxBArea - interArea)

    # enclosed_x1 = min(boxA[0], boxA[0])
    # enclosed_y1 = min(boxA[1], boxA[1])
    # enclosed_x2 = max(boxA[2], boxA[2])
    # enclosed_y2 = max(boxA[3], boxA[3])
    # enclosed_area = (enclosed_x2 - enclosed_x1) * (enclosed_y2 - enclosed_y1)
    # enclosed_area = enclosed_area.to(torch.float32)
    #
    # # Compute GIoU
    # giou = iou - ((enclosed_area - (boxAArea + boxBArea)) / enclosed_area)

    # Return the IOU
    return iou


def get_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:  # No overlap
        return 0
    i = w_intersection * h_intersection
    u = w1 * h1 + w2 * h2 - i  # Union = Total Area - Intersection Area
    return i/u


def plot_dynamic_graph(d, value, batch_no):
    d(value, batch_no)


def convert_to_yolo_full_scale(bbox, grid_id):
    real_x = bbox[0] * ((int(grid_id / 7) + 1) * 64)
    real_y = bbox[1] * ((int(grid_id) % 7 + 1) * 64)
    real_w = bbox[2] * 448
    real_h = bbox[3] * 448
    return [real_x, real_y, real_w, real_h]


def convert_to_box_coordinates(x, y, w, h, grid_id):
    bbox = []

    real_x = x * ((int(grid_id / 7) + 1) * 64)
    real_y = y * ((int(grid_id) % 7 + 1) * 64)
    real_w = w * 448
    real_h = h * 448

    # calculate x and y coordinates of the top-left corner of the rectangle
    x1 = real_x - real_w / 2
    y1 = real_y - real_h / 2
    bbox.append(np.round(x1))
    bbox.append(np.round(y1))

    # calculate x and y coordinates of the top-right corner of the rectangle
    x2 = real_x + real_w / 2
    y2 = real_y - real_h / 2
    bbox.append(np.round(x2))
    bbox.append(np.round(y2))

    # calculate x and y coordinates of the bottom-right corner of the rectangle
    x3 = real_x + real_w / 2
    y3 = real_y + real_h / 2
    bbox.append(np.round(x3))
    bbox.append(np.round(y3))

    # calculate x and y coordinates of the bottom-left corner of the rectangle
    x4 = real_x - real_w / 2
    y4 = real_y + real_h / 2
    bbox.append(np.round(x4))
    bbox.append(np.round(y4))

    return bbox


def draw_all_bboxes_from_annotations(annt):
    annt = torch.reshape(annt, (49, 6))
    img = ImageElement()
    grid_id = 0
    for elem in annt:
        if elem[0] > 0.4:
            bbox = convert_to_box_coordinates(elem[1], elem[2], elem[3], elem[4], grid_id)
            img.set_bbox(bbox)
            img.draw_box()
        grid_id += 1


def image_checks(image, size_x, size_y) -> np.ndarray:
    # Some images from the dataset are greyscale, so they need to be
    # converted to RGB before placing them as input in the network.
    if image.shape == (size_x, size_y):
        image = grey2rgb(image)

    # Retain only RGB values from images with an Alpha channel
    if image.shape == (size_x, size_y, 4):
        image = image[:, :, 0:3]

    # Normalize image to have pixel values in [0,1] interval
    if np.max(image) - np.min(image) != 0:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    else:
        return np.zeros(1)

    return image


def torch_prepare(image, annotations) -> tuple:
    transform = tv.Compose([tv.ToTensor()])
    image = transform(image)
    image = torch.reshape(image, (1, 3, dim, -1))
    annotations = dataset.AerialImagesDataset.no_args_construct().build_grids_annotations(annotations)
    annotations = annotations.astype(np.float16)
    annotations = transform(annotations)
    image = image.to(torch.device('cuda'))
    annotations = annotations.reshape(1, 49 * (5 + no_of_classes))
    return image, annotations


def conv_yolo_2_dota(bbox):
    """
    Returns the coordinates of all four vertices of a rectangle given its center and dimensions.
    """
    x, y = bbox[0], bbox[1]  # center point coordinates
    w_half, h_half = bbox[2] / 2, bbox[3] / 2  # half width and half height
    # calculate the four vertices
    top_left_x, top_left_y = (x - w_half, y - h_half)
    top_right_x, top_right_y = (x + w_half, y - h_half)
    bottom_right_x, bottom_right_y = (x + w_half, y + h_half)
    bottom_left_x, bottom_left_y = (x - w_half, y + h_half)
    return [top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y,
            bottom_left_x, bottom_right_y]


init()
