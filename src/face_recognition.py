import cv2
import joblib
import numpy as np
from tkinter import Tk, filedialog

from face_detector import FaceDetector
from face_extractor import FaceExtractorAndEncoding


class FaceRecognitionApp:
    def __init__(self, model_path):
        self.detector = FaceDetector(device="CPU:0")
        self.extractor = FaceExtractorAndEncoding(device="CPU")

        print("Loading SVM face recognition model...")
        data = joblib.load(model_path)
        self.svm = data["model"]
        self.label_encoder = data["labels"]

    def predict_from_image(self, image):
        faces = self.detector.detect_bounding_box_on_frame_video(image)

        if not faces:
            return []

        embeddings = self.extractor.extract_and_encode(image, faces)

        results = []
        for item in embeddings:
            emb = item["embedding"].reshape(1, -1)

            pred = self.svm.predict(emb)[0]
            proba = self.svm.predict_proba(emb).max()

            name = self.label_encoder.inverse_transform([pred])[0]

            results.append({
                "name": name,
                "confidence": proba,
                "box": item["box"]
            })

        return results

    def run_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise SystemExit("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.predict_from_image(frame)

            for r in results:
                x, y, w, h = r["box"]
                label = f"{r['name']} ({r['confidence']:.2f})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Face Recognition - Webcam", frame)

            if cv2.waitKey(1) == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_image_dialog(self):
        root = Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )

        if not file_path:
            return

        image = cv2.imread(file_path)
        results = self.predict_from_image(image)

        for r in results:
            x, y, w, h = r["box"]
            label = f"{r['name']} ({r['confidence']:.2f})"

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Face Recognition - Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
