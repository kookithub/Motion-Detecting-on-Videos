import cv2
from ultralytics import YOLO

VIDEO = r"D:\XXX\XXX.mp4"
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 2.5

FRAME_STEP = int(fps)  # ~1 frame per second
frame_idx = 0
hits = []
i = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_STEP == 0:
        results = model(frame, conf=0.3, classes=[0], verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            t = frame_idx / fps
            hits.append(t)
    if (frame_idx %431 == 0):
        print(i)
        i+=1
    frame_idx += 1

cap.release()

for t in hits:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    print(f"{h:02d}:{m:02d}:{s:02d}")
