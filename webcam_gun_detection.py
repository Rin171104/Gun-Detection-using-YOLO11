import cv2
from ultralytics import YOLO

# Load model
model = YOLO(
    r"D:\HocPython\deep learning\Computer Vision\gun detection\runs_gun\gun_detector_tb\weights\best.pt"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Không mở được webcam")

print("🎥 Webcam started. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6, imgsz=640, device=0)[0]
    output = results.plot()

    # ===== CẢNH BÁO =====
    if len(results.boxes) > 0:
        # nền đỏ
        cv2.rectangle(output, (0, 0), (output.shape[1], 120), (0, 0, 255), -1)

        cv2.putText(
            output,
            "⚠️ WARNING: GUN DETECTED!",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            3
        )

    cv2.imshow("Gun Detection - Webcam", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
