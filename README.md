# ObjectDetection

  This project is an implementation of YOLOv1 model (https://arxiv.org/pdf/1506.02640.pdf), with some adaptations for aerial images detection. It was deigned to be trained on DOTA dataset, with features like image cropping and text files processing.
  
  Requirements:
  
- torch 1.13+cu117
- torchvision 0.14.0
- scikit-learn 1.2.1
- numpy 1.23.5
- matplotlib 3.6.2
- PyYAML 6.0
- pandas 1.5.2  

Steps for correct pre-processing DOTA data:

- setup the paths, class labels and the field showOneImage to <b>false</b> in configs/pre-processing-config.yaml
- run "python showImageFromDataset.py"
- run "python convertToYOLOcsv.py"

Steps for training a model on DOTA data:

- setup the training configs: paths, hyperparamters and checkpoint to <b>false</b> (training can be done with checkpoints because the best and last weights are saved)
- setup general configs with preffered values
- run "python training.py"

Steps for testing a trained model:

- setup the testing configs: paths
- for an image from DOTA dataset, set oneDatasetImage to <b>true</b>. The ground truth can also be shown by setting draw_ground_truth.
- for testing the whole test set to get performance metrics, set all boolean fields to <b>false</b>
- for testing the model on a random image, set oneRandomImage to <b>true</b>
- setup general configs with the same ones on which the model was trained
- run "python testing.py"

Here are some examples of detections from trained models:

![image](https://github.com/edinitu/ObjectDetection/assets/80175654/9ed96f36-1262-4b69-9f9f-3e530cd54c7b)

![image](https://github.com/edinitu/ObjectDetection/assets/80175654/dec79f73-b5c6-433b-aa49-f2767f38f9af)

![image](https://github.com/edinitu/ObjectDetection/assets/80175654/3e14dae1-d2b8-4448-9733-14c83fb8a306)






