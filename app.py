from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import time
import threading
import socket
import requests
import RPi.GPIO as GPIO
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# === INIT ===
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ML Model
model_path = 'models/poacher_detector_model.h5'
model = load_model(model_path)

# PIR GPIO Setup
PIR_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

# Telegram Bot
BOT_TOKEN ='your bot token'
CHAT_ID = 'chat id'

# === VIDEO INFERENCE FUNCTION ===
def process_video(video_path, model):
    IMAGE_SIZE = (224, 224)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized = cv2.resize(face, IMAGE_SIZE)
            input_data = np.expand_dims(resized / 255.0, axis=0)
            prediction = model.predict(input_data, verbose=0)[0][0]
            if prediction > 0.5:
                detected_frames.append(frame_count)
            break
        frame_count += 1
    cap.release()
    return detected_frames

# === ALERT FUNCTION ===
def send_poacher_alert_sms():
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'text': '🚨 Poacher Detected for Over 5 Seconds! Please Take Action.'
    }
    requests.post(url, data=payload)
    print("✅ SMS Alert Sent via Telegram!")

# === RECORD CAMERA VIDEO ===
def record_video_from_camera(duration_sec=10, output_path="uploads/motion_capture.mp4"):
    print(f"🎥 Recording {duration_sec}s video...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    start_time = time.time()
    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Video saved to {output_path}")
    return output_path

# === HANDLE MOTION ===
def motion_detected_handler():
    print("🔔 PIR Motion Detected.")
    video_path = record_video_from_camera()
    if not video_path or not os.path.exists(video_path):
        print("❌ Video recording failed.")
        return

    detected_frames = process_video(video_path, model)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    poacher_duration = len(detected_frames) / fps
    print(f"🕵️ Poacher visible for {poacher_duration:.2f}s in video.")

    if poacher_duration > 5:
        send_poacher_alert_sms()
    else:
        print("ℹ️ No alert. Poacher not detected long enough.")

# === PIR SENSOR LISTENER ===
def monitor_pir_sensor():
    print("📡 Monitoring PIR sensor on GPIO 23...")
    try:
        while True:
            if GPIO.input(PIR_PIN):
                motion_detected_handler()
                time.sleep(12)  # Wait before allowing another detection
            else:
                print("🕊️ No motion detected.")
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("❌ PIR monitoring stopped.")

# === OPTIONAL: SOCKET SERVER ===
def handle_connection(conn, addr):
    print(f"📥 Socket connection from {addr}")
    data = conn.recv(1024)
    if b"MOTION" in data:
        motion_detected_handler()
    conn.close()

def start_socket_server():
    HOST = '0.0.0.0'
    PORT = 9999
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(5)
    print(f"🔌 Socket server running on port {PORT}")
    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle_connection, args=(conn, addr)).start()

# === FLASK API (OPTIONAL) ===
@app.route('/')
def home():
    return "✅ Poacher Detection API running."

# === MAIN ENTRY ===
if __name__ == '__main__':
    threading.Thread(target=monitor_pir_sensor).start()
    threading.Thread(target=start_socket_server).start()
    app.run(host='0.0.0.0', port=5000)
