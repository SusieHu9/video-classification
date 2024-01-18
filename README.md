# README
# Multi-class Sport Video Classification: Transfer Learning & Moving Averages

## Introduction:
This project aims to construct a multi-class video classifier for 5 distinct sports activities, utilizing a fusion of `transfer learning` and `moving averages` with `Python` and `Keras`. Throughout the development of this video classifier, comprehensive steps were taken, including `data exploration`, `pre-processing`, and `image augmentation` techniques facilitated by `OpenCV`. The focus then shifted to transfer learning for image classification, leveraging pre-trained models (`ResNet50`, `EfficientNetB0`, `VGG16`) for feature extraction, and retraining the last fully connected layer for improved output. To address the temporal nature of videos, the image classifier was retrained with additional data, and moving averages were applied to derive the average prediction from multiple frames. The project evaluates model performance using various metrics, including the Confusion Matrix, Precision, Recall, Accuracy, and F1 score, crucial for obtaining a more reliable prediction.

### Why is it original?
The conventional method for video classification typically involves training an RNN-CNN model. However, this approach presents challenges, including the demand for substantial computational resources, practical difficulties, and scalability limitations. Consequently, the key innovation introduced in this project lies in the utilization of moving averages over predictions of multiple frames for each video. This innovative approach effectively tackles the challenges associated with RNN-CNN models and flickering, resulting in a more stable prediction for each video.

### Model Results:
For image classification, transfer learning using VGG16 as the pre-trained model performs the best with the highest precision (0.86), recall (0.86), accuracy (0.86), and f1 score (0.86) on the test set. ResNet50 ranks second and finally EfficientNetB0.

For video classification, transfer learning with VGG16 as the pre-trained model demonstrates superior performance, resulting in a 12% increase in model accuracy. Following closely is ResNet50, and lastly, EfficientNetB0. The hierarchy of model performance in video classification mirrors that observed in image classification.

## Getting Started
### Prerequisite:
```bash
pip install Pillow
pip install pandas
pip install numpy
pip install opencv-python
pip install tensorflow==2.9.1
pip install scikit-learn
pip install matplotlib
```

### Run the notebook
To run the notebook, follow these steps:
1. Clone the repository to your local machine and Use `Google Colab` to run the notebook `notebook.ipynb`
2. Create a new folder named `PDF Project` in `My Drive` of Google Drive
3. Upload the image and video resources from folders, including `data/Sport Images` and `data/Sport Videos`
4. Run the notebook to get the results


## Project Development
### Data Source: 
- data/Sport Images/ImageURLs: image data source

### File Description:
- notebook.ipynb: solution codes
- data/Sport Images: image data (input data)
- data/Sport Videos: video data (input data)
- data/test: sample test data (code output)
- data/train: sample train data (code output)
- data/validation: sample validation data (code output)

