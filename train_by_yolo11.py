from ultralytics import YOLO
import multiprocessing as mp

def main():
    model = YOLO("models/yolo11n.pt")

    model.train(
        data="gun_data.yaml",
        epochs=15,
        imgsz=640,
        batch=8,
        device=0,
        workers=6,
        project="runs_gun",
        name="gun_detector_tb",
        exist_ok=True,
        plots=True
    )

    print(" Training finished")

if __name__ == "__main__":
    mp.freeze_support()
    main()
