import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import cv2

from utils import dataset_dir


class FaceSVMPipeline:
    dataset_directory = dataset_dir
    def __init__(self, detector, extractor):
        self.detector = detector
        self.extractor = extractor
        
        self.label_encoder = LabelEncoder()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", probability=True))
        ])

    def load_dataset_embeddings(self):
        X = []
        y = []

        for person_name in sorted(os.listdir(self.dataset_directory)):
            person_dir = os.path.join(self.dataset_directory, person_name)

            if not os.path.isdir(person_dir):
                continue

            print(f"Processing person: {person_name}")

            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(person_dir, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                faces = self.detector.detect_bounding_box_on_image_array(image)

                if not faces:
                    continue

                embeddings = self.extractor.extract_and_encode(image, faces)

                for item in embeddings:
                    X.append(item["embedding"])
                    y.append(person_name)

        return np.array(X), np.array(y)

    def train(self):
        X, y = self.load_dataset_embeddings()

        if len(X) == 0:
            raise RuntimeError(
                "No faces detected in dataset. Check face detection."
            )

        y_encoded = self.label_encoder.fit_transform(y)

        print("Training SVM classifier...")
        self.model.fit(X, y_encoded)


    def save(self, model_path="svm_face_model.pkl"):
        joblib.dump({
            "model": self.model,
            "labels": self.label_encoder
        }, model_path)

        print(f"SVM model saved to {model_path}")
