from mtcnn import MTCNN
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
import matplotlib.pyplot as plt
from mtcnn.utils.plotting import plot
import cv2

class FaceDetector: 

    def __init__(self, device="CPU:0"):
        self.detector = MTCNN(device=device)
        self.frame = None
        self.faces = []


    def detect_bounding_box_on_image(self,image_path):
        detector = self.detector

        image = load_image(image_path)

        result = detector.detect_faces(image)

        return result

    def draw_detections(self,image,result):
        plt.imshow(plot(image, result))
        plt.show()


    def detect_bounding_box_on_frame_video(self, frame):
        if frame is None:
            return []

        # IMPORTANT: Convertir BGR (OpenCV) en RGB (MTCNN)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            faces = self.detector.detect_faces(rgb_frame)
            
            if not faces:
                return []

            for face in faces:
                x, y, w, h = face["box"]
                conf = face.get("confidence", 0)

                # Dessiner sur le 'frame' original (qui est en BGR pour l'affichage)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            return faces

        except Exception as e:
            print(f"Erreur lors de la détection : {e}")
            return []


    def startup_camera_and_analyze(self):
       
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise SystemExit("Cannot open webcam")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Cannot receive frame")
                break
            self.frame = frame.copy()
            self.faces = self.detect_bounding_box_on_frame_video(frame)

            cv2.imshow("Webcam Detection", frame)

            if cv2.waitKey(1) == 27:  # ESC pour quitter
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_bounding_box_on_image_array(self, image):

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_image)
        return faces
    