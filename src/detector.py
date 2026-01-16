import cv2
from ultralytics import YOLO

class VisionDetector:
    def __init__(self, model_name="yolov8n-seg.pt"):
        print(f"[INFO] loading {model_name} model...")
        self.model = YOLO(model_name)

        self.line_y = 750
        self.counter = 0
        self.already_counted = set()

    def run_inference(self, image):
        results = self.model.track(image, persist=True, stream=True, verbose=False)
        return results

    def detect_video(self, source_path):
        cap = cv2.VideoCapture(source_path)

        if not cap.isOpened():
            print(f"[INFO] file {source_path} not opened")
            return

        target_width = 540
        target_height = 960

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

                        if center_y > self.line_y and obj_id not in self.already_counted:
                            self.counter += 1
                            self.already_counted.add(obj_id)

            cv2.line(annotated_frame, (0, self.line_y), (int(cap.get(3)), self.line_y), (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Summ of objects: {self.counter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("VisionFlow - Smart Counter", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = VisionDetector()
    detector.detect_video("data/short_video.mp4")