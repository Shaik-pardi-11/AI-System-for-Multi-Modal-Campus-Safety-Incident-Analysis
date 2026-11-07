
import cv2
import time
from collections import Counter
from ultralytics import YOLO

# Phone 1 live camera
CAMERA_URL = "http://100.75.50.187:8080/video"

# Lightweight YOLO26 Nano model
MODEL_NAME = "yolo26n.pt"

print("Loading YOLO26n...")
model = YOLO(MODEL_NAME)

print("Connecting to Phone 1...")
cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    raise RuntimeError("Cannot connect to Phone 1")

print("Camera connected. Starting detection...")

fps_start = time.time()
frame_count = 0
fps = 0

while True:
    success, frame = cap.read()

    if not success:
        print("Frame unavailable. Retrying...")
        time.sleep(0.1)
        continue

    # Resize for faster inference
    frame = cv2.resize(frame, (960, 540))

    # YOLO object detection
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.35,
        verbose=False
    )

    result = results[0]

    # Annotated frame
    annotated = result.plot()

    # Count detected objects
    counts = Counter()

    if result.boxes is not None:
        for cls in result.boxes.cls.cpu().numpy():
            name = model.names[int(cls)]
            counts[name] += 1

    people = counts.get("person", 0)
    vehicles = sum(
        counts.get(name, 0)
        for name in ["car", "motorcycle", "bus", "truck"]
    )

    frame_count += 1
    elapsed = time.time() - fps_start

    if elapsed >= 1:
        fps = frame_count / elapsed
        fps_start = time.time()
        frame_count = 0

    # Dashboard overlay
    cv2.rectangle(annotated, (10, 10), (440, 115), (25, 25, 55), -1)

    cv2.putText(
        annotated,
        f"PEOPLE: {people}  VEHICLES: {vehicles}",
        (22, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}  CAMERA: PHONE 1",
        (22, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 255, 180),
        2
    )

    cv2.imshow("I-Mage | Live YOLO26 Detection", annotated)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
