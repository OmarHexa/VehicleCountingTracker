import cv2
import math
import numpy as np
import time
import random
from ultralytics import YOLO
from ultralytics.utils.plotting import *
from sort import Sort
import cvzone

class VehicleCounter:
    def __init__(self, video_path: str, resolution: tuple[int,int],line: list[int], mask_path: str|None = None):
        self.video_path = video_path
        self.resolution = resolution
        self.mask = mask
        self.line = line
        self.model = YOLO('./Yolo-weights/yolov8n.pt')
        self.model.fuse()
        self.classNames = self.model.names
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.totalcount = []
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

    def calculate_fps(self, prev_time, new_time):
        return 1 / (new_time - prev_time)

    def process_frame(self, img, mask):
        img = cv2.resize(img, self.resolution)
        mask = cv2.resize(mask, self.resolution)
        img_region = cv2.bitwise_and(img, mask)
        return img_region, img

    def calculate_dimensions(self, x1, y1, x2, y2):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        return x1, y1, x2, y2, w, h

    def perform_detection(self, img):
        results = self.model(img, stream=True)
        detections = np.empty((0, 5))
        transform = self.calculate_dimensions # temporary function handle
        for r in results:
            boxes = r.boxes
            # masks = r.masks
            for box in boxes:
                x1, y1, x2, y2, _, _ = transform(*box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                currentClass = self.classNames[int(box.cls[0])]

                if currentClass in ["car", "truck", "motorbike", "bus"] and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        return detections

    def perform_counting(self, img):
        results_tracker = self.tracker.update(self.detections)
        for result in results_tracker:
            *_, track_id = result
            x1, y1, x2, y2, w, h = self.calculate_dimensions(*result[:-1])
            cv2.rectangle(img, (x1, y1), (x2, y2), (self.colors[int(track_id) % len(self.colors)]), 3)
            cvzone.putTextRect(img, f' {int(track_id)}', (max(0, x1), max(35, y1)), colorR=(128, 128, 128),
                               scale=2, thickness=3, offset=10)
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if self.line[0] < cx < self.line[2] and self.line[1] - 10 < cy < self.line[1] + 30:
                if track_id not in self.totalcount:
                    self.totalcount.append(track_id)
                    cv2.line(img, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 255, 0), 3)
        return img

    def count_vehicles(self):
        video = cv2.VideoCapture(self.video_path)

        prev_frame_time = 0
        new_frame_time = 0

        while True:
            new_frame_time = time.time()

            success, img = video.read()
            if not success:
                print("Video has ended or no frame found.")
                break

            masked_img, img = self.process_frame(img, self.mask)
            self.detections = self.perform_detection(masked_img)

            img = self.perform_counting(img)

            cv2.line(img, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 0, 255), 3)

            cvzone.putTextRect(img, f' Count: {len(self.totalcount)}', (50, 50), colorR=(128, 128, 128),
                            scale=2, thickness=3, offset=10)

            fps = self.calculate_fps(prev_frame_time, new_frame_time)
            print(f"FPS:{fps}")
            prev_frame_time = new_frame_time

            cv2.imshow("Image", img)

            # Handle key events properly
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage:
    video_path = "./videos/cars.mp4"
    mask_path = "./assets/mask.png"
    resolution = (1280, 720)
    mask = cv2.imread(mask_path)
    line = (0, 400, 1280, 400)

    vehicle_counter = VehicleCounter(video_path, resolution, line,mask)
    vehicle_counter.count_vehicles()
