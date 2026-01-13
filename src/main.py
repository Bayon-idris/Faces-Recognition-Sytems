from face_recognition import FaceRecognitionApp

def main():
    app = FaceRecognitionApp(model_path="svm_face_model.pkl")

    print("1 - Webcam recognition")
    print("2 - Image recognition")

    choice = input("Choose mode (1/2): ")

    if choice == "1":
        app.run_webcam()
    elif choice == "2":
        app.run_image_dialog()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()

