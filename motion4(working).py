import cv2
from ultralytics import YOLO

VIDEO_PATH = "D:/XXX/XXX.mp4"
CONF_THRESH = 0.4
FRAME_SKIP = 3          # increase if video is very long

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0
person_active = False   # tracks if a person is currently present

print("START")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    results = model(frame, conf=CONF_THRESH, verbose=False)

    person_detected = False

    if results[0].boxes is not None:
        # class 0 = person
        if (results[0].boxes.cls == 0).any():
            person_detected = True

    # ✅ PERSON JUST APPEARED
    if person_detected and not person_active:
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        sec = int(ms / 1000)

        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60

        print(f"PERSON APPEARED AT {h:02d}:{m:02d}:{s:02d}")
        person_active = True

    # ✅ PERSON IS GONE — reset state
    elif not person_detected and person_active:
        person_active = False

cap.release()
print("DONE")
