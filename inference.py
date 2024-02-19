import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Path to your video file
video_path = "./videos/cars.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

# Get the original video's width and height
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the desired resolution for resizing
desired_width = 640
desired_height = 480


def show(cap):
    while True:
    # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            print("Video has ended.")
            break

        # Resize the frame to the desired resolution
        resized_frame = cv2.resize(frame, (desired_width, desired_height))

        # Display the resized frame
        cv2.imshow("Resized Video", resized_frame)

        # Wait for a key event (press 'q' to exit)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
show(cap)
# Release the video capture object and destroy all windows
cap.release()
# out.release()  # Uncomment if saving the resized video
cv2.destroyAllWindows()
