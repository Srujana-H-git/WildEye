import cv2
import numpy as np
import time
import os
import RPi.GPIO as GPIO
from tensorflow.keras.models import load_model
from picamera2 import Picamera2

# === Constants ===
IMAGE_SIZE = (224, 224)
MODEL_PATH = 'poacher_detector_model.h5'
BUZZER_PIN = 17  # You can change this to any available GPIO pin

# === Load Model ===
model = load_model(MODEL_PATH)

# === Setup GPIO for Buzzer ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)  # Initially off

# === Initialize Camera ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("Camera started. Press 'q' or Ctrl+C to stop.")

try:
    while True:
        frame = picam2.capture_array()
        input_frame = cv2.resize(frame, IMAGE_SIZE)
        input_frame = np.expand_dims(input_frame / 255.0, axis=0)

        # Make prediction
        prediction = model.predict(input_frame, verbose=0)[0][0]
        if prediction > 0.5:
            label = "Poacher"
            color = (0, 0, 255)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn buzzer ON
        else:
            label = "No Poacher"
            color = (0, 255, 0)
            GPIO.output(BUZZER_PIN, GPIO.LOW)   # Turn buzzer OFF

        # Display label
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.imshow("Poacher Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    print("Cleaning up...")
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.close()
