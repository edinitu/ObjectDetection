import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image

import utils


class ImageElement:
    """
        Python class for displaying an image from DOTA dataset with corresponding bounding boxes.
    """
    def __init__(self):
        self.bounding_box = []
        self.label = ''
        self.number_label = 0
        self.image_metadata = ImageMetadata()
        self.confidence = 0
        self.yolo_bbox = []

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

    def get_label(self):
        return self.label

    def get_number_label(self):
        return self.number_label

    def get_image_metadata(self):
        return self.image_metadata

    def set_bbox(self, bbox):
        self.bounding_box = bbox

    def set_confidence(self, confidence):
        self.confidence = confidence

    def get_confidence(self):
        return self.confidence

    def set_yolo_bbox(self, yolo_bbox):
        self.yolo_bbox = yolo_bbox

    def get_yolo_bbox(self):
        return self.yolo_bbox

    def convert_yolo_to_dota(self, grid_id):
        self.bounding_box = utils.convert_to_box_coordinates(
            self.yolo_bbox[0],
            self.yolo_bbox[1],
            self.yolo_bbox[2],
            self.yolo_bbox[3],
            grid_id
        )

    def draw_box(self, color='black'):
        x = []
        y = []
        for i in range(len(self.bounding_box)):
            if i % 2 == 0:
                x.append(self.bounding_box[i])
            else:
                y.append(self.bounding_box[i])
        plt.plot(x[0:2], y[0:2], color=color, label=self.label)
        plt.plot(x[1:3], y[1:3], color=color)
        plt.plot(x[2:4], y[2:4], color=color)
        plt.plot([x[3], x[0]], [y[3], y[0]], color=color)


class ImageMetadata:
    def __init__(self):
        self.image_source = ''
        self.gsd = ''

    def set_image_source(self, img_source):
        self.image_source = img_source

    def set_gsd(self, gsd):
        self.gsd = gsd

    def get_image_source(self):
        return self.image_source

    def get_gsd(self):
        return self.gsd


def write_line(file, part, bbox):
    for coord in bbox:
        file.write(str(coord) + " ")
    file.write(part.get_label() + " ")
    file.write(part.get_number_label())
    file.write('\n')


def write_to_txt(path, part, bbox):
    if not os.path.exists(path):
        with open(path, 'a+') as f:
            f.write('imagesource:'+part.get_image_metadata().get_image_source()+'\n')
            f.write('gsd:' + part.get_image_metadata().get_gsd()+'\n')
            write_line(f, part, bbox)
    else:
        with open(path, 'a+') as f:
            write_line(f, part, bbox)


# TODO Comment this algorithm
def crop(cropped_path, txt_path, input, height, width, image_parts, labels, max_objects, k):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    bbox_in_imgbox = False
    new_bboxes = {}
    bboxes_for_one_piece = []
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j+width, i+height)
            objects_in_one_piece = 0
            for part in image_parts:
                if part.get_label() not in labels:
                    continue
                if objects_in_one_piece > max_objects:
                    break
                if part.get_bounding_box()[0] >= j and part.get_bounding_box()[1] >= i\
                        and part.get_bounding_box()[4] <= j+width and part.get_bounding_box()[5] <= i+height:

                    bboxes_for_one_piece.append(convert_bbox_to_smaller_image(part.get_bounding_box(), j, i))
                    bbox_to_write = convert_bbox_to_smaller_image(part.get_bounding_box(), j, i)
                    path = os.path.join(txt_path, input[len(input)-9:len(input)-4] + "-" + str(k) + ".txt")
                    write_to_txt(path, part, bbox_to_write)
                    objects_in_one_piece += 1
                    bbox_in_imgbox = True

            # save only images that contain minimum 1 bounding box for training
            if bbox_in_imgbox:
                a = im.crop(box)
                a.save(os.path.join(cropped_path, input[len(input)-9:len(input)-4] + "-%s.png" % k))
                bbox_in_imgbox = False

                new_bboxes[k] = bboxes_for_one_piece.copy()
                bboxes_for_one_piece.clear()
                k += 1
    return new_bboxes


# TODO Comment this algorithm
def read_one_image_labels(img_label, labels):
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
            if img_element.get_label() not in labels:
                continue
            img_element.set_number_label(str(labels[line[8]]))

            img_elements.append(img_element)
    return img_elements


def convert_bbox_to_smaller_image(bbox, j, i):
    new_bbox = []
    t = 0
    for point in bbox:
        if t % 2 == 0:
            new_bbox.append(point - j)
        else:
            new_bbox.append(point - i)
        t += 1
    return new_bbox


if __name__ == '__main__':
    # load necessary configs from yaml file
    with open('configs/pre-processing-config.yaml') as f:
        config = yaml.safe_load(f)

    pre_process_cfg = config['processImages']
    images = pre_process_cfg['images']
    txt_labels = pre_process_cfg['txt_labels']
    labels = pre_process_cfg['labels']
    option = pre_process_cfg['show_one_img']
    crop_dim = pre_process_cfg['cropped_img_dim']
    max_objects = pre_process_cfg['max_objects_in_piece']

    #   If we set the option in the config to show only one image, then the script will
    # plot that image with its bounding boxes drawn and also (for demo purposes) another
    # plot with a 'piece' of set dimension cropped from the original image with its bounding
    # boxes. If the option is not set, then the script will automatically crop all images
    # from a folder, compute the new bounding boxes coordinates and load them in 2 different
    # folders: one for the new cropped images and one for the new annotations.
    if option:
        item = pre_process_cfg['img_to_show']
        image_cropped_path = pre_process_cfg['cropped_imgs']
        elements = read_one_image_labels(os.path.join(txt_labels, item + '.txt'), labels)
        image_path = os.path.join(images, item + '.png')
        img = plt.imread(image_path)

        plt.figure()
        # Get original image bounding boxes and show image with them.
        image_parts = []
        for element in elements:
            image_parts.append(element)
            element.draw_box()

        plt.imshow(img)
        plt.plot()

        new_txt_annotations = pre_process_cfg['cropped_labels']
        # Get new bboxes for cropped parts of original image
        k = 0
        new_bboxes = crop(
            image_cropped_path,
            new_txt_annotations,
            image_path,
            crop_dim,
            crop_dim,
            image_parts,
            labels,
            max_objects,
            k)

        # show one "cut" of the bigger image with its bounding boxes
        image_piece_number = str(pre_process_cfg['cropped_img_piece'])
        cropped_imgs_path = os.path.join(image_cropped_path, item + '-' + image_piece_number + '.png')
        img = plt.imread(cropped_imgs_path)
        plt.figure()

        for i in range(len(new_bboxes[int(image_piece_number)])):
            elements[i].set_bbox(new_bboxes[int(image_piece_number)][i])
            elements[i].draw_box()

        plt.imshow(img)
        plt.plot()
        plt.show(block=True)

    else:
        image_cropped_path = pre_process_cfg['cropped_imgs']
        new_txt_annotations = pre_process_cfg['cropped_labels']
        k = 0
        for filename in os.listdir(images):
            elements = read_one_image_labels(os.path.join(txt_labels, filename.strip('.png') + '.txt'), labels)
            if not elements:
                continue
            image_path = os.path.join(images, filename)
            img = plt.imread(image_path)

            # Get all image elements from the original image: bounding boxes and labels.
            image_parts = []
            for element in elements:
                image_parts.append(element)

            # Crop image into crop_dim x crop_dim parts, compute new bounding boxes coordinates and save them to
            # new txt files.
            new_bboxes = crop(
                image_cropped_path,
                new_txt_annotations,
                image_path,
                crop_dim,
                crop_dim,
                image_parts,
                labels,
                max_objects,
                k)
