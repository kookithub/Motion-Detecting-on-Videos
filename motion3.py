import cv2
import os
import numpy as np

VIDEO = "D:/XXX/XXX.mp4"
OUT = "previews"
os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

prev = None
last_event = -999
EVENT_GAP = 3  # seconds

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    if prev is not None:
        diff = cv2.absdiff(prev, gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        motion_pixels = cv2.countNonZero(mask)

        if motion_pixels > 1200:
            t = frame_idx / fps

            if t - last_event > EVENT_GAP:
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)

                ts = f"{h:02d}:{m:02d}:{s:02d}"
                filename = f"{OUT}/{ts.replace(':','_')}.jpg"

                cv2.imwrite(filename, frame)
                print("Motion:", ts)

                last_event = t

    prev = gray
    frame_idx += 1

cap.release()
