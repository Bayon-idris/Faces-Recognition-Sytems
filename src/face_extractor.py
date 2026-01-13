import cv2
import os
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch


class FaceExtractorAndEncoding:
    
    def __init__(self, device="CPU"):
        self.device = torch.device("cpu")

        print("Loading FaceNet model for embeddings")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)


    def crop_face_from_box(self, image, box):
        
        x, y, w, h = box

        x = max(0, x)
        y = max(0, y)

        face = image[y:y+h, x:x+w]

        if face.size == 0:
            raise ValueError("Cropped face is empty")

        return face


    def encode_face_to_embedding(self, face_image):
        """
        Convert a face image to a numerical vector (embedding)
        """
        face_resized = cv2.resize(face_image, (160, 160))

        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        face_rgb = face_rgb.astype(np.float32) / 255.0

        tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.cpu().numpy().flatten()

    # -----------------------------------------------------

    def extract_and_encode(self, frame, mtcnn_faces):
        
        embeddings = []

        for face in mtcnn_faces:
            box = face["box"]

            face_crop = self.crop_face_from_box(frame, box)

            emb = self.encode_face_to_embedding(face_crop)

            embeddings.append(
                {
                    "embedding": emb,
                    "confidence": face.get("confidence", 0),
                    "box": box
                }
            )

        return embeddings
