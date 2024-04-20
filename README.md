# Face Mask Detection with Temperature Monitoring

This project aims to detect whether a person is wearing a face mask and monitor their temperature using a webcam. The system displays real-time video with bounding boxes around faces and labels indicating whether the person is wearing a mask or not. Additionally, it provides information about the person's temperature and health status.

## Requirements
- Python 3
- OpenCV
- TensorFlow
- Keras
- imutils
- Flask

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/face-mask-detection.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the following command to start the application:
```bash
python detect_mask_video.py
```
This will start a Flask server and open a webpage with the live video feed showing face mask detection and temperature monitoring.

## Acknowledgments
Face detection model: prototxt and pre-trained model from OpenCV.  
Face mask detection model: Trained using TensorFlow and Keras.
