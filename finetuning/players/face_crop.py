import cv2
import numpy as np

def crop_to_face(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Expand the box slightly to include hair/shoulders if needed
        margin = int(0.3 * h)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        cropped = img[y1:y2, x1:x2]
        # Resize the cropped image to 200x200
        cropped = cv2.resize(cropped, (200, 200))
        
        # Convert to RGBA (add alpha channel)
        rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        
        # Create mask for green screen
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        
        # Define green color range in HSV
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Create mask where green is found
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Invert mask and apply to alpha channel
        rgba[:, :, 3] = cv2.bitwise_not(mask)
        
        # Save with transparency (use PNG format)
        cv2.imwrite(output_path, rgba)
        break  # only crop first face

crop_to_face("out-0.png", "cropped.png")
