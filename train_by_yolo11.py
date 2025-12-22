from ultralytics import YOLO
import multiprocessing as mp

def main():
    # Load YOLO11 pretrained model
    model = YOLO("models/yolo11n.pt")

    # Train detection model
    model.train(
        data="gun_data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project="runs_gun",
        name="gun_detector",
        patience=20
    )

    print("✅ TRAINING COMPLETED")

if __name__ == "__main__":
    mp.freeze_support()
    main()
