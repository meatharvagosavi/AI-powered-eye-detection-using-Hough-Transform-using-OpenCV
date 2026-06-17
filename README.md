# 👁️ AI-Powered Eye Detection using Hough Transform

This project combines **Artificial Intelligence (Haar Cascade Classifiers)** with the **Circular Hough Transform** to detect human eyes and locate the iris/pupil region in an image.

## 🚀 Features

* Face detection using Haar Cascade classifier
* Eye detection within detected faces
* Noise reduction using Median Blur
* Iris/Pupil detection using Circular Hough Transform
* Visualization of detected circles and centers

## 🛠️ Technologies Used

* Python
* OpenCV (`cv2`)
* NumPy

## 📂 Project Workflow

1. Load pre-trained Haar Cascade classifiers for face and eye detection.
2. Read the input image and convert it to grayscale.
3. Detect faces in the image.
4. Detect eyes within each detected face.
5. Apply median filtering to reduce noise.
6. Use Circular Hough Transform to detect circular iris/pupil boundaries.
7. Draw detected circles and centers on the image.

## 📸 Output Visualization

* **Green Circle:** Detected iris/pupil boundary
* **Red Dot:** Center of the detected circle

## 📦 Installation

Install the required libraries:

```bash
pip install opencv-python numpy
```

## ▶️ Run the Project

Replace:

```python
image = cv2.imread('sample_face.jpg')
```

with the path to your image and run:

```bash
python eye_detection.py
```

## 📁 Required Files

* `eye_detection.py`
* Input image (e.g., `sample_face.jpg`)

## 🎯 Applications

* Eye tracking systems
* Driver drowsiness detection
* Biometric authentication
* Human-computer interaction
* Medical image analysis

## 📚 References

* OpenCV Documentation
* Hough Circle Transform
* Haar Cascade Classifiers

## ⭐ Future Improvements

* Real-time webcam-based eye tracking
* Deep learning-based eye detection
* Gaze estimation and blink detection

---

Made with ❤️ using OpenCV and Python.
