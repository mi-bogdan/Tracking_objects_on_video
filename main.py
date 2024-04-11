import cv2
import time
import numpy as np


# Загрузка предварительно обученной модели
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel"
)

# Классы объектов
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


video_source = 0
cap = cv2.VideoCapture(video_source)

# Инициализация трекера и переменных для FPS
tracker = None
fps_tracker = cv2.legacy.TrackerMedianFlow_create()
fps_start_time = None
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if fps_start_time is None:
        fps_start_time = time.time()

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

            # Инициализация или обновление трекера
            if tracker is None or not tracker.update(frame)[0]:
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(frame, (startX, startY, endX - startX, endY - startY))

    # Расчет и вывод FPS
    fps_end_time = time.time()
    fps = frame_counter / (fps_end_time - fps_start_time)

    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Оптический поток Farneback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_counter == 1:
        prev_gray = gray
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
    else:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Optical Flow", rgb)
        prev_gray = gray

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
