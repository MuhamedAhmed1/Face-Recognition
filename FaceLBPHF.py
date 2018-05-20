import cv2
import cv2.face
import os
import numpy as np

subjects = ["","User1","User2","User3","User4","User5","User6","User7","User8","User9","User10","User11","User12","User13","User14","User15","User16","User17","User18","User19","User20","User21","User22","User23","User24","User25","User26","Mohamed Ahmed"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("User"):
            continue;

        label = int(dir_name.replace("User", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)

            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("test-data")
print("Data prepared")


print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

def detect_video_faces():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_scaleFactor = 1.2
    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, face_scaleFactor, 5)

        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            label, confidence = face_recognizer.predict(roi_gray)

            label_text = subjects[label]

            cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

        cv2.imshow('img', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
    cap.release()

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

print("Predicting images...")

detect_video_faces()

print("Prediction complete")

cv2.waitKey(0)
cv2.destroyAllWindows()