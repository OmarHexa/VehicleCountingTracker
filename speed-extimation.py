from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2

cap = cv2.VideoCapture("./videos/cars.mp4")
mask_area = cv2.imread("./assets/mask.png")
assert cap.isOpened(), "Error reading video file"
detectionLine = [(190, 500), (1086, 500)]
model = YOLO('./Yolo-weights/yolov8n-seg.pt')
names = model.model.names
model.fuse()

# Init speed-estimation obj
speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=detectionLine,
                   names=names,
                   view_img=True)

while cap.isOpened():

    success, im0 = cap.read()
    im0 = cv2.resize(im0,(1280,720))
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0, tracks)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
