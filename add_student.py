import cv2
import os
# Load the Haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to detect face and crop it
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:  # ✅ Correct way to check
        return None
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]

# Ask for student name and create folder
name = input("Enter Student Name: ")
path = f'dataset/{name}'
os.makedirs(path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    face = face_extractor(frame)
    if face is not None:
        count += 1
        face = cv2.resize(face, (200, 200))
        file_name_path = f'{path}/{count}.jpg'
        cv2.imwrite(file_name_path, face)

        # Show counter on screen
        cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Face Cropper - Press Enter to Exit', frame)
    if cv2.waitKey(1) == 13 or count == 50:  # 13 = Enter key
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Images Collected and Saved in:", path)
