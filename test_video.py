import cv2
from ultralytics import YOLO

model = YOLO(
    r"D:\HocPython\deep learning\Computer Vision\gun detection\runs_gun\gun_detector\weights\best.pt"
)
cap = cv2.VideoCapture("video/gun_video1.mp4")
assert cap.isOpened(), "❌ Không mở được video input"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "gun_detection_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect gun
    results = model(frame, conf=0.5)[0]
    output_frame = results.plot()

    # 👉 HIỂN THỊ
    cv2.imshow("Gun Detection", output_frame)

    # 👉 LƯU VIDEO
    out.write(output_frame)

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# 5. RELEASE
# ===============================
cap.release()
out.release()
cv2.destroyAllWindows()

