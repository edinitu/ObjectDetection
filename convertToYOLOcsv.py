import csv
import os
import yaml


class Annotation:
    def __init__(self, clazz, x_center, y_center, width, height):
        self.__clazz = clazz
        self.__x_center = x_center
        self.__y_center = y_center
        self.__width = width
        self.__height = height

    @property
    def clazz(self):
        return self.__clazz

    @property
    def x_center(self):
        return self.__x_center

    @property
    def y_center(self):
        return self.__y_center

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height


def convertToYOLO(bbox):
    yolo_labels = []
    x_midpoint = (float(bbox[0])+float(bbox[4]))/2
    y_midpoint = (float(bbox[1])+float(bbox[7]))/2

    height = abs(float(bbox[1])-float(bbox[7]))
    width = abs(float(bbox[0])-float(bbox[2]))

    yolo_labels.append(x_midpoint)
    yolo_labels.append(y_midpoint)
    yolo_labels.append(width)
    yolo_labels.append(height)

    return yolo_labels


if __name__ == '__main__':
    with open('configs/pre-processing-config.yaml') as f:
        config = yaml.safe_load(f)

    convertYOLO_cfg = config['convertYOLO']
    txt_folder = convertYOLO_cfg['txt']
    csv_folder = convertYOLO_cfg['csv']

    annotations_list = []

    for filename in os.listdir(txt_folder):
        f = os.path.join(txt_folder, filename)
        if os.path.isfile(f):
            with open(f, 'r') as file:
                file.readline()
                file.readline()
                while True:
                    line = file.readline().split(' ')
                    if line == ['']:
                        break
                    bbox = line[0:8]
                    yolo_format = [int(line[9].replace('\n', ''))]
                    yolo_format.extend(convertToYOLO(bbox))

                    annotation = Annotation(yolo_format[0], yolo_format[1],
                                            yolo_format[2], yolo_format[3], yolo_format[4])
                    annotations_list.append(
                        [annotation.clazz, annotation.x_center, annotation.y_center, annotation.width,
                         annotation.height]
                    )

            csvfilename = filename.replace('.txt', '.csv')
            with open(os.path.join(csv_folder, csvfilename),
                      'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(annotations_list)
                annotations_list.clear()
