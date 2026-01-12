from face_detector import FaceDetector
from face_extractor import FaceExtractorAndEncoding
from svm_trainer import FaceSVMPipeline

def main():
    detector = FaceDetector(device="CPU:0")
    extractor = FaceExtractorAndEncoding(device="CPU")

    svm_pipeline = FaceSVMPipeline(
        detector=detector,
        extractor=extractor
    )

    svm_pipeline.train()
    svm_pipeline.save("svm_face_model.pkl")

if __name__ == "__main__":
    main()
