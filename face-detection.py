import numpy as np
import cv2
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torch

# Initialize the neural network
mtcnn = MTCNN(keep_all=True, device='cpu')

# Initiliaze the OpenCV video capture
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('Cannot open the camera')

while True:
    # Read an image from the camera
    ret, frame = capture.read()
    if not ret:
        print('Cannot receive a frame')
        break
        
    # Convert the image from OpenCV format to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect faces
    boxes, _ = mtcnn.detect(image)

    # Draw boxes around faces
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    draw.text((0, 0), 'Press Q to quit', (255, 255, 255))

    # Display the image
    cv2.imshow('Detected faces', cv2.cvtColor(np.array(image_with_boxes), cv2.COLOR_RGB2BGR))
    
    # Quit if Q is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
capture.release()
# Close the window
cv2.destroyAllWindows()
