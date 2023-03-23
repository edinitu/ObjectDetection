# ObjectDetection
This project contains:

  - <b>showImageFromDataset</b>: a script that can be used either to show an image from the dataset with its corresponding bounding boxes 
  or to crop all images from dataset given specific dimensions and load them in a folder. It also loads in a folder the new txt 
  annotations for the cropped images.
  
  - <b>convertToYOLO</b>: a script that converts the txt annotations which contain coordinates for the 4 points defining the bounding box
  to csv files with YOLO format rows.
  
  - <b>customDataset</b>: defines a new Dataset based on the Dataset from torch.utils.data. Prepares the dataset to be iterated at training time.
  
  - <b>model</b>: Class for the network's final model - 24 convolutional layers + 2 fully connected layers.
  
  - <b>training</b>: script for training the model
  
  - <b>testing</b>: script for testing the model on 1 image, or on the whole testing set
  
  - <b>metrics</b> utility classes used for computing relevant metrics such as recall or average precision
  
  - <b>utils</b>: useful functions to help in other scripts
  
  - <b>tests</b>: unit tests
