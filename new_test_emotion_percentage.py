import cv2
import numpy as np
from keras.models import model_from_json
import time

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.weights.h5")
print("Loaded model from disk")

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize emotion counts and start time
emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        
        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        predicted_emotion = emotion_dict[maxindex]
        emotion_counts[predicted_emotion] += 1
    
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate the session duration
end_time = time.time()
session_duration = end_time - start_time

# Calculate the percentage of each emotion
total_frames = sum(emotion_counts.values())
emotion_percentages = {emotion: (count / total_frames * 100 if total_frames > 0 else 0) for emotion, count in emotion_counts.items()}

# Write the report to a file
with open("emotion_report.txt", "w") as file:
    file.write(f"Total Session Duration: {session_duration:.2f} seconds\n")
    file.write("Emotion Percentages:\n")
    for emotion, percentage in emotion_percentages.items():
        file.write(f"{emotion}: {percentage:.2f}%\n")

# Display the report in a separate window
report_window = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.putText(report_window, f"Total Session Duration: {session_duration:.2f} seconds", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
y_offset = 100
for emotion, percentage in emotion_percentages.items():
    cv2.putText(report_window, f"{emotion}: {percentage:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    y_offset += 50

cv2.imshow('Emotion Report', report_window)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed

# Clean up
cap.release()
cv2.destroyAllWindows()
