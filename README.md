Here's a README file for your project:

---

# Real-Time Sign Language Recognition System

This project is a real-time hand gesture recognition system designed to translate sign language into text using a trained neural network model. It includes data collection, keypoint extraction, model training, and real-time prediction.

## Project Overview

1. **Data Collection** - Uses Mediapipe and OpenCV to capture frames and extract keypoints for each gesture.
2. **Keypoint Extraction** - Processes images to identify hand landmarks for consistent input to the model.
3. **Model Training** - Trains a Convolutional-LSTM model on the collected keypoints.
4. **Real-Time Prediction** - Captures live video, processes frames, and uses the model to predict gestures.

## Folder Structure

- `KeypointsExtraction.py`: Extracts and draws keypoints from each frame.
- `DataCollection.py`: Collects gestures based on the selected action.
- `ModelTraining.py`: Trains and saves the neural network model.
- `realtime.py`: Runs the real-time gesture prediction using the trained model.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- Mediapipe
- Numpy
- Keyboard (for capturing gestures with 'Spacebar')


## Usage

### Data Collection
1. Define the gesture name in `DataCollection.py` (replace `'X'` with the desired gesture names seperated by commas).
2. Run `DataCollection.py` to start capturing data for the selected gesture.
3. Press the `Spacebar` to start recording each sequence and `q` to quit.

### Model Training
1. After collecting data, run `ModelTraining.py` to train the model.
2. The trained model will be saved as `my_model.h5`.

### Real-Time Prediction
1. Run `realtime.py` to start real-time gesture recognition.
2. The program will display predictions in a sentence format on the video feed.

## Demonstration Video and Documents

- **Video Demonstration**: https://youtu.be/1Gtk4cD07s4 (youtube link)
- **Research Paper and Poster**: Available in the documentation folder.