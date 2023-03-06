import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import torch
from showImageFromDataset import ImageElement

plt.ion()


class DynamicUpdate(Thread):
    """
    Class for plotting training statistics: loss per batch, average processing time per batch etc.
    """
    # Suppose we know the x range
    min_x = 0
    #max_x = 250     # number of batches

    def __init__(self, title, max_x=250):
        super().__init__()
        self.title = title
        self.max_x = max_x
        self.on_launch()
        self.xdata = []
        self.ydata = []

    def on_launch(self):
        # Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], 'o')
        self.ax.set_title(self.title)
        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        # Other stuff
        self.ax.grid()
        ...

    def on_running(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # Example
    def __call__(self, value, batch_no):
        self.xdata.append(batch_no)
        self.ydata.append(value)
        self.on_running(self.xdata, self.ydata)

        return self.xdata, self.ydata

# TODO Define list of PredictionStats objects and pass it to AveragePrecision to compute it


class FinalPredictions:
    # TODO This should be configurable
    no_of_grids = 49

    def __init__(self, outputs, truth):
        self.grids = {}
        self.truth = {}
        grid_id = 0
        outputs = torch.reshape(outputs, (49, 6))
        truth = torch.reshape(outputs, (49, 6))
        count = 0
        for elem in truth:
            if elem[0] == 1:
                img_elem = ImageElement()
                img_elem.set_yolo_bbox([elem[1], elem[2], elem[3], elem[4]])
                img_elem.set_label('plane')
                self.truth[count] = img_elem
            count += 1

        for elem in outputs:
            if elem[0] < 0.3:
                grid_id += 1
                continue
            img_elem = ImageElement()
            img_elem.set_yolo_bbox([elem[1], elem[2], elem[3], elem[4]])
            img_elem.set_confidence(elem[0])
            if elem[5] > 0.5:
                img_elem.set_label('plane')
            self.grids[grid_id] = img_elem
            grid_id += 1

        self.non_max_suppression()
        self.convert_to_dota()

    # TODO Define unit test for this
    def non_max_suppression(self):
        conf_id_map = {}
        for key in self.grids.keys():
            conf_id_map[self.grids[key].get_confidence()] = key

        sorted_conf = []
        for key in conf_id_map.keys():
            sorted_conf.append(key)
        sorted_conf.sort(reverse=True)

        for conf in sorted_conf:
            if conf_id_map[conf] not in self.grids:
                continue
            self.remove_overlapped_boxes(conf_id_map[conf])

    def remove_overlapped_boxes(self, reference_key):
        for key in range(self.no_of_grids):
            if key != reference_key and key in self.grids:
                iou = get_iou(self.grids[reference_key].get_yolo_bbox(), self.grids[key].get_yolo_bbox())
                if iou > 0.4:
                    del self.grids[key]

    def add_to_stats_list(self):
        # TODO Populate a list of confidences and confusions
        pass

    def convert_to_dota(self):
        for key in self.grids.keys():
            self.grids[key].convert_yolo_to_dota(key)

    def draw_boxes(self):
        for elem in self.grids.values():
            elem.draw_box()

    def get_grids(self):
        return self.grids


def grey2rgb(img):
    assert img.shape == (448, 448)
    new_img = np.zeros((448, 448, 3), dtype=np.float16)
    new_img[:, :, 0] = img
    new_img[:, :, 1] = img
    new_img[:, :, 2] = img
    return new_img


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


# def conv_yolo_2_dota(bbox):
#     """
#     Returns the coordinates of all four vertices of a rectangle given its center and dimensions.
#     """
#     x, y = bbox[0], bbox[1]  # center point coordinates
#     w_half, h_half = bbox[2] / 2, bbox[3] / 2  # half width and half height
#     # calculate the four vertices
#     top_left_x, top_left_y = (x - w_half, y - h_half)
#     top_right_x, top_right_y = (x + w_half, y - h_half)
#     bottom_right_x, bottom_right_y = (x + w_half, y + h_half)
#     bottom_left_x, bottom_left_y = (x - w_half, y + h_half)
#     return [top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y,
#             bottom_left_x, bottom_right_y]

