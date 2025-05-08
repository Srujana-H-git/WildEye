import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === PATHS ===
train_data_dir = 'poacher_dataset'  # should have 'poacher' and 'no_poacher' folders
test_video_path = 'TEST3.mp4'
output_video_path = 'templates/output4_vid.mp4'
model_path = 'models/poacher_detector_model.h5'

# === IMAGE PARAMETERS ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# === 1. DATA GENERATOR (with Augmentation and Validation Split) ===
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# === 2. MODEL DEFINITION ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# === 3. COMPILE MODEL ===
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# === 4. TRAIN MODEL ===
model.fit(train_generator, validation_data=val_generator, epochs=15)

# === 5. SAVE MODEL ===
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"Model saved to {model_path}")

# === 6. VIDEO PROCESSING FUNCTION ===
def process_video(video_path, model, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    poacher_detected_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, IMAGE_SIZE)
        input_frame = np.expand_dims(resized_frame / 255.0, axis=0)

        prediction = model.predict(input_frame, verbose=0)[0][0]
        label = "Poacher" if prediction > 0.5 else "No Poacher"
        color = (0, 0, 255) if label == "Poacher" else (0, 255, 0)

        if label == "Poacher":
            poacher_detected_frames.append(frame_count)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return poacher_detected_frames

# === 7. RUN VIDEO ANALYSIS ===
if os.path.exists(test_video_path):
    print(f"Processing test video: {test_video_path}")
    model = load_model(model_path)
    frames = process_video(test_video_path, model, output_video_path)
    print(f"Poachers detected in {len(frames)} frames")
    print(f"Processed video saved to {output_video_path}")
else:
    print("Test video not found.")
