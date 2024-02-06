from ultralytics import YOLO
from ultralytics.utils.plotting import *
import cv2
import math
import time
import numpy as np
from sort import Sort  # Assuming 'sort.py' contains the implementation of the SORT tracker
import cvzone
import random


WIDTH = 1280
HIGHT = 720

def calculate_fps(prev_time, new_time):
    return 1 / (new_time - prev_time)

def process_frame(img, mask, resolution):
    # Resize the frame and the mask to the desired resolution
    img = cv2.resize(img, resolution)
    mask = cv2.resize(mask, resolution)

    # Apply a bitwise AND operation to combine the frame and the mask
    imgRegion = cv2.bitwise_and(img, mask)
    return imgRegion, img

def calculate_dimensions(x1, y1, x2, y2):
    # Convert coordinates to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Calculate width and height
    w, h = x2 - x1, y2 - y1
    return x1, y1, x2, y2, w, h


def perform_detection(img, model, classNames):
    # Perform object detection using YOLO on the entire image
    results = model(img, stream=True)
    # annotate = Annotator(img,line_width=1,font_size=2)
    # Initialize an empty array for detections
    detections = np.empty((0, 5))

    # Process YOLO results
    for r in results:
        boxes = r.boxes
        masks = r.masks
        for box,mask in zip(boxes,masks):
            # Extract bounding box coordinates
            x1, y1, x2, y2,_,_ = calculate_dimensions(*box.xyxy[0])
            # Extract confidence and class information
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # annotate.box_label(box.xyxy[0],f"{currentClass}")
            # annotate.draw_centroid_and_tracks(box.xyxy[0], color=(255, 0, 255), track_thickness=2)

            # Check if the detected object is a specific class and confidence is above a threshold
            if currentClass in ["car", "truck", "motorbike", "bus"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                # img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255), 3)
                detections = np.vstack((detections, currentArray))

    return detections



def perform_counting(img, tracker, detections,limits,totalcount,colors):
    # Update the SORT tracker with the detections
    resultsTracker = tracker.update(detections)
    # Process and display the tracked results
    for result in resultsTracker:
        *_, id = result
        x1,y1,x2,y2,w,h = calculate_dimensions(*result[:-1])
        # Draw a rectangle around the tracked object
        # cvzone.cornerRect(img, (x1, y1, w, h), t=2, l=5, colorC=(0, 0, 255), colorR=(0, 255, 0))
        cv2.rectangle(img,(x1,y1),(x2,y2),(colors[int(id) % len(colors)]), 3)

        # Display the object ID near the tracked object
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), colorR=(128, 128, 128),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx< limits[2] and  limits[1]-10 <cy <limits[1]+30:
            if totalcount.count(id) == 0:
                totalcount.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),3)
    return None

def count_vehicles(video,line,resolution):

     # Load the YOLO model
    model = YOLO('./Yolo-weights/yolov8n-seg.pt')
    model.fuse()
    # Initialize the SORT tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # co-ordinates of the count line

    # variable to store car count
    totalcount = []
    # Initialize variables for frame timing
    prev_frame_time = 0
    new_frame_time = 0
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    # Main loop for video processing
    while True:
        new_frame_time = time.time()

        # Read a frame from the video
        success, img = video.read()
        if not success:
            print("Video has ended or no frame found.")
            break

        masked_img, img = process_frame(img,mask,resolution)

        # Perform object detection using YOLO on the masked region
        detections = perform_detection(masked_img, model=model,classNames=classNames)

        # Perform object tracking and update the image
        perform_counting(img, tracker, detections,line,totalcount,colors)
        

        cv2.line(img,(detectionLine[0],detectionLine[1]),(detectionLine[2],detectionLine[3]),(0,0,255),3)

        cvzone.putTextRect(img, f' Count: {len(totalcount)}', (50,50), colorR=(128, 128, 128),
                            scale=2, thickness=3, offset=10)
        # Calculate and print the frames per second (fps)
        fps = calculate_fps(prev_frame_time,new_frame_time)
        print(f"FPS:{fps}")
        prev_frame_time = new_frame_time
        # Display the frame with annotations
        cv2.imshow("Image", img)

        # Wait for a key event and check if 'q' key is pressed to break the loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

if __name__ == "__main__":
    # Open the video file for reading
    cap = cv2.VideoCapture("./videos/cars.mp4")
    mask = cv2.imread("./assets/mask.png")


    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]

    # Set the desired resolution
    resolution = (WIDTH, HIGHT)
    detectionLine = [190, 701, 1086, 701]

    count_vehicles(cap,detectionLine,resolution)
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
