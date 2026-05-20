import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Load trained model and label map
model = cv2.face.LBPHFaceRecognizer_create()
model.read('trainer/trained_model.yml')
label_map = np.load('trainer/labels.npy', allow_pickle=True).item()

# Load Haar cascade classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Face detection function
def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return (img,)
    for (x, y, w, h) in faces:
        return (img, gray[y:y+h, x:x+w], x, y, w, h)

# Ask for expected name
expected_name = input("👤 Enter expected student name (e.g. sanjivani): ").strip().lower()

# Start webcam
cap = cv2.VideoCapture(0)
attendance = []

print("\n📸 Press 'Enter' to stop and save attendance")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    result = face_detector(frame)

    if len(result) == 6:
        image, face, x, y, w, h = result
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)
        name = label_map[label].lower()

        if confidence < 60:
            if name == expected_name:
                now = datetime.now()
                time_str = now.strftime("%H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")
                attendance.append([name, time_str, date_str])

                cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Wrong Person!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"⚠️ Face detected is NOT '{expected_name}'. Detected: {name}")
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition - Attendance", frame)

    if cv2.waitKey(1) == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()

# Save attendance
os.makedirs("Attendance", exist_ok=True)

if len(attendance) > 0:
    df = pd.DataFrame(attendance, columns=["Name", "Time", "Date"])
    df.drop_duplicates(subset=["Name", "Date"], keep="last", inplace=True)

    # Save today's attendance
    filename = f"Attendance/Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(filename, index=False)
    print("✅ Attendance saved to:", filename)

    # Update master file
    master_file = "Attendance/All_Attendance.csv"
    if os.path.exists(master_file):
        df_master = pd.read_csv(master_file)
        df_combined = pd.concat([df_master, df])
        df_combined.drop_duplicates(subset=["Name", "Date"], keep="last", inplace=True)
        df_combined.to_csv(master_file, index=False)
    else:
        df.to_csv(master_file, index=False)

    print("📘 All_Attendance.csv updated.")

else:
    print("⚠️ No attendance saved. Wrong person or no match.")
