from mtcnn import MTCNN
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
import matplotlib.pyplot as plt
from mtcnn.utils.plotting import plot

class FaceDetector: 

    def __init__(self, device="CPU:0"):
        self.detector = MTCNN(device=device)

    def detect_face_on_image(self,image):
        detector = self.detector

        image = load_image("360_F_243123463_zTooub557xEWABDLk0jJklDyLSGl2jrr.jpg")

        result = detector.detect_faces(image)

        return result

    def draw_detections(self,image,result):
        plt.imshow(plot(image, result))
        plt.show()
