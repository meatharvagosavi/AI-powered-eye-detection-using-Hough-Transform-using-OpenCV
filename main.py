import cv2
import numpy as np

# 1. Load the pre-trained AI classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 2. Read the image and convert to grayscale
# Replace 'sample_face.jpg' with the path to your image
image = cv2.imread('sample_face.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. AI Detection: Find the face
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    # Extract the Region of Interest (ROI) for the face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    # AI Detection: Find the eyes within the face ROI
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    for (ex, ey, ew, eh) in eyes:
        # Extract the ROI specifically for the eye
        eye_region_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_region_color = roi_color[ey:ey+eh, ex:ex+ew]
        
        # 4. Pre-processing: Apply median blur to reduce noise (like eyelashes)
        blurred_eye = cv2.medianBlur(eye_region_gray, 5)
        
        # 5. Circular Hough Transform
        # param1: upper threshold for the internal Canny edge detector
        # param2: accumulator threshold (lower = more false circles detected)
        circles = cv2.HoughCircles(
            blurred_eye, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20,
            param1=50, 
            param2=30, 
            minRadius=5, 
            maxRadius=30
        )
        
        # 6. Draw the detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle (Iris/Pupil boundary) in Green
                cv2.circle(eye_region_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle in Red
                cv2.circle(eye_region_color, (i[0], i[1]), 2, (0, 0, 255), 3)

# Display the final output
cv2.imshow('AI & Hough Transform Eye Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
