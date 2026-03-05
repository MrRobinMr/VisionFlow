import os
import cv2
from ultralytics import YOLO

# For better detection: use a detection model (not seg) and larger size – e.g. yolov8m.pt or yolov8l.pt
DEFAULT_MODEL = "yolov8m.pt"


class VisionDetector:
    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        conf=0.4,
        iou=0.5,
        imgsz=640,
        tracker="bytetrack.yaml",
    ):
        """
        Args:
            model_name: YOLO model. Use detection (.pt, e.g. yolov8m.pt) for counting;
                larger (m/l/x) = better accuracy, slower.
            conf: Confidence threshold (0.25–0.6). Higher = fewer false positives, may miss faint objects.
            iou: NMS IoU threshold. Lower = fewer overlapping boxes.
            imgsz: Inference size. Larger (e.g. 1280) = better small-object detection, slower.
            tracker: Tracker config (bytetrack.yaml, botsort.yaml).
        """
        print(f"[INFO] loading {model_name} model...")
        self.model = YOLO(model_name)

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = tracker

        self.line_y = 750
        self.counter = 0
        self.already_counted = set()
        self.count_by_class = {}

        # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [0, 2, 3, 5, 7]

    def run_inference(self, image):
        results = self.model.track(
            image,
            persist=True,
            stream=True,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=self.tracker,
            classes=self.target_classes,
        )
        return results

    def detect_video(
        self,
        source_path,
        output_name="output.mp4",
        save_report=True,
        scale=0.5,
        line_zone_height=30,
    ):
        """
        scale: Resize factor for display/inference (0.25=fast, 0.5=balanced, 1.0=best detection, slower).
        line_zone_height: Pixels used as counting zone; larger = less chance to miss fast objects.
        """
        cap = cv2.VideoCapture(source_path)

        if not os.path.exists("results"):
            os.makedirs("results")

        if not cap.isOpened():
            print(f"[INFO] file {source_path} not opened")
            return None

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_width = int(orig_width * scale)
        target_height = int(orig_height * scale)

        self.line_y = int(target_height * 0.8)
        self.line_zone_height = line_zone_height

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_count = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = f"results/{output_name}"
        out = cv2.VideoWriter(out_path, fourcc, fps, (target_width, target_height))

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            results = self.run_inference(frame)

            for r in results:
                annotated_frame = r.plot()

                if r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.int().cpu().tolist()
                    clss = r.boxes.cls.int().cpu().tolist()

                    for box, obj_id, cls_id in zip(boxes, ids, clss):
                        x1, y1, x2, y2 = box
                        center_y = int((y1 + y2) / 2)

                        if self.line_y < center_y < (self.line_y + self.line_zone_height):
                            if obj_id not in self.already_counted:
                                self.counter += 1
                                self.already_counted.add(obj_id)
                                self.count_by_class[cls_id] = self.count_by_class.get(cls_id, 0) + 1

            frame_count += 1

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (target_width, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

            cv2.line(annotated_frame, (0, self.line_y), (target_width, self.line_y), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"OBJECTS: {self.counter}", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

            out.write(annotated_frame)

            cv2.imshow("VisionFlow - Smart Counter", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        duration_sec = frame_count / fps if fps else 0
        stats = {
            "source_path": source_path,
            "output_path": out_path,
            "total_count": self.counter,
            "count_by_class": dict(self.count_by_class),
            "duration_sec": duration_sec,
            "fps": fps,
            "frame_count": frame_count,
        }
        if save_report:
            from report import generate_report
            generate_report(
                source_path=source_path,
                output_path=out_path,
                total_count=self.counter,
                count_by_class=self.count_by_class,
                duration_sec=duration_sec,
                fps=fps,
                frame_count=frame_count,
            )
        return stats


if __name__ == "__main__":
    # Default: balanced speed/accuracy (yolov8m, conf=0.4, scale=0.5)
    detector = VisionDetector()

    # For best detection (slower): larger model, higher resolution, larger imgsz
    # detector = VisionDetector(model_name="yolov8l.pt", conf=0.35, imgsz=1280)
    # detector.detect_video("data/high-angle.mp4", scale=0.75, line_zone_height=40)

    #detector.detect_video("data/high-angle.mp4")
    detector.detect_video("data/short_video.mp4")