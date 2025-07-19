import cv2
import numpy as np
import dlib
from imutils import face_utils
import streamlit as st
from twilio.rest import Client
import datetime
import face_recognition
import os

# Twilio credentials
account_sid = 'AC8f6c9c8b19845d2663283a573ddc173b'
auth_token = 'b37452a400e608be664e04c4ca4ab3ae'
twilio_phone_number = '+12549876773'
target_phone_number = '+919817219535'
client = Client(account_sid, auth_token)

# Initialize Streamlit
st.title("Driver Drowsiness Detection")
start_button = st.button("Start Detection")
stframe = st.empty()

# Load dlib face detector and shape predictor
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

sleep = drowsy = active = 0
status = ""
color = (0, 0, 0)

# Load known face encodings
known_faces = []
known_names = []
training_image_folder = 'Training_images'

for filename in os.listdir(training_image_folder):
    image_path = os.path.join(training_image_folder, filename)
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_faces.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])

def recognize_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_faces, encoding)
        if True in matches:
            index = matches.index(True)
            return known_names[index]
    return "Unknown"

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    return 0

def send_sms(message):
    try:
        client.messages.create(
            to=target_phone_number,
            from_=twilio_phone_number,
            body=message
        )
    except Exception as e:
        print(f"SMS Error: {e}")

def play_alert(alert_type):
    try:
        beep_path = "/Users/kunikabhadra/Downloads/PRJ/beep.mp3"
        if os.path.exists(beep_path):
            os.system(f'afplay "{beep_path}"')
        else:
            print(f"Beep file not found at {beep_path}")
    except Exception as e:
        print(f"Sound Error: {e}")

def save_status(name, status):
    if status in ["SLEEPING !!!", "Drowsy !"]:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('Status.txt', 'a') as f:
            f.write(f"{now}: {name} - {status}\n")

# Start detection loop
while start_button:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                play_alert("sleeping")
                send_sms("Driver detected sleeping!")
                name = recognize_face(frame)
                save_status(name, status)
        elif left_blink == 1 or right_blink == 1:
            sleep = active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                play_alert("drowsy")
                send_sms("Driver detected drowsy!")
                name = recognize_face(frame)
                save_status(name, status)
        else:
            drowsy = sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    stframe.image(frame, channels="BGR")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
