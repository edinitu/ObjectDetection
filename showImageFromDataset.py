import os
import matplotlib.pyplot as plt

'''
    Python class for displaying an image from DOTA dataset with correspoding bounding boxes.
'''
class ImageElement:
    def __init__(self):
        self.bounding_box = []
        self.label = ''
        self.number_label = 0
        self.image_metadata = ImageMetadata()

    def set_bounding_box(self, coordinates):
        for coordinate in coordinates:
            self.bounding_box.append(float(coordinate))

    def set_label(self, label):
        self.label = label

    def set_number_label(self, number_label):
        self.number_label = number_label

    def set_image_metadata(self, image_metadata):
        self.image_metadata = image_metadata

    def get_bounding_box(self):
        return self.bounding_box

    def draw_box(self):
        x = []
        y = []
        for i in range(len(self.bounding_box)):
            if i % 2 == 0:
                x.append(self.bounding_box[i])
            else:
                y.append(self.bounding_box[i])
        plt.plot(x[0:2], y[0:2], color='black', label=self.label)
        plt.plot(x[1:3], y[1:3], color='black')
        plt.plot(x[2:4], y[2:4], color='black')
        plt.plot([x[3], x[0]], [y[3], y[0]], color='black')


class ImageMetadata:
    def __init__(self):
        self.image_source = ''
        self.gsd = ''

    def set_image_source(self, img_source):
        self.image_source = img_source

    def set_gsd(self, gsd):
        self.gsd = gsd


def read_one_image_labels(img_label):
    with open(img_label, 'r') as file:
        img_elements = []
        img_data = ImageMetadata()
        img_data.set_image_source(file.readline().split(':')[1].strip('\n'))
        img_data.set_gsd(file.readline().split(':')[1].strip('\n'))
        while True:
            img_element = ImageElement()
            img_element.set_image_metadata(img_data)
            line = file.readline().split(' ')
            if line == ['']:
                break
            img_element.set_bounding_box(line[0:8])
            img_element.set_label(line[8])
            img_element.set_number_label(line[9].strip('\n'))
            img_elements.append(img_element)
    return img_elements


TRAIN_DATA_PATH = input('Enter train images root directory path: ')
TRAIN_LABELS_PATH = input('Enter root directory for txt annotations: ')

item = input('Enter image to show: ')
elements = read_one_image_labels(os.path.join(TRAIN_LABELS_PATH, item + '.txt'))
img = plt.imread(os.path.join(TRAIN_DATA_PATH, item +'.png'))

plt.figure()
for element in elements:
    element.draw_box()

plt.imshow(img)
plt.plot()
plt.show()
