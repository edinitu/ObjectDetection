import numpy as np
import matplotlib.pyplot as plt
from threading import Thread

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


def grey2rgb(img):
    assert img.shape == (448, 448)
    new_img = np.zeros((448, 448, 3), dtype=np.float32)
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

