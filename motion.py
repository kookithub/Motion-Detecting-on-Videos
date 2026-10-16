import cv2

VIDEO = "D:/XXX/XXX.mp4"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

prev = None
motion_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔴 EXACT timestamp mask (from your image)
    gray[0:120, 1200:2047] = 0

    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev is None:
        prev = gray
        continue

    diff = cv2.absdiff(prev, gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    REAL_MOTION = False
    for c in contours:
        if cv2.contourArea(c) > 6000:  # filters text & noise
            REAL_MOTION = True
            break

    if REAL_MOTION:
        t = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        motion_times.append(t)

    prev = gray

cap.release()

print("REAL motion timestamps (seconds):")
print(motion_times)
