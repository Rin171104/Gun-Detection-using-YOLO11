import cv2
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Test YOLO gun detection on video )")
    parser.add_argument("--path","-p",type=str,required=True,help="Path to input video")
    parser.add_argument("--output",type=str,default="gun_detection_output.mp4",help="Path to output video")
    parser.add_argument("--conf",type=float,default=0.5,help="Confidence threshold")
    parser.add_argument("--imgsz",type=int,default=640,help="Inference image size")
    parser.add_argument("--device",type=str,default="0",help="Device: 0 for GPU, cpu for CPU")
    parser.add_argument("--show",action="store_true",help="Show video while processing")
    return parser.parse_args()


def main(args):
    model = YOLO(r"D:\HocPython\deep learning\Computer Vision\gun detection\runs_gun\gun_detector_tb\weights\best.pt")
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        raise IOError(f" Cannot open video: {args.path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device
        )[0]

        output_frame = results.plot()
        writer.write(output_frame)

        if args.show:
            cv2.imshow("Gun Detection", output_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f" Done. Output saved to: {args.output}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
