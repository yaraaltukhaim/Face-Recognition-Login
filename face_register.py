import cv2
import face_recognition
import numpy as np
import os

if not os.path.exists("registered_faces"):
    os.makedirs("registered_faces")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if face_locations:
        cv2.imwrite("registered_faces/user1.jpg", frame)
        print("Face Captured!")
        break

cap.release()
cv2.destroyAllWindows()

# Convert face to an encoding
image = face_recognition.load_image_file("registered_faces/user1.jpg")
encoding = face_recognition.face_encodings(image)[0]
np.save("registered_faces/user1.npy", encoding)
print("Face Encoding Saved!")
