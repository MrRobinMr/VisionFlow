import os
import cv2
from ultralytics import YOLO

class VisionDetector:
    def __init__(self, model_name="yolov8s-seg.pt"):
        print(f"[INFO] loading {model_name} model...")
        self.model = YOLO(model_name)

        self.line_y = 750
        self.counter = 0
        self.already_counted = set()

        # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [0, 2, 3, 5, 7]

    def run_inference(self, image):
        results = self.model.track(
            image,
            persist=True,
            stream=True,
            verbose=False,
            conf=0.35,
            classes=self.target_classes
        )
        return results

    def detect_video(self, source_path, output_name="output.mp4"):
        cap = cv2.VideoCapture(source_path)

        if not os.path.exists("results"):
            os.makedirs("results")

        if not cap.isOpened():
            print(f"[INFO] file {source_path} not opened")
            return

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_width = orig_width // 4
        target_height = orig_height // 4

        self.line_y = int(target_height * 0.8)

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"results/{output_name}", fourcc, fps, (target_width, target_height))

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

                    for box, obj_id in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        center_y = int((y1 + y2) / 2)

                        if self.line_y < center_y < (self.line_y + 20):
                            if obj_id not in self.already_counted:
                                self.counter += 1
                                self.already_counted.add(obj_id)

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


if __name__ == "__main__":
    detector = VisionDetector()
    detector.detect_video("data/short_video.mp4")
    #detector.detect_video("data/high-angle.mp4")