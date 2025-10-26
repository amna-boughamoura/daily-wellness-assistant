import cv2
import time
import csv
from datetime import datetime
from deepface import DeepFace

# 🎥 Setup webcam
cap = None
for i in range(3):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        print(f"✅ Found working camera at index {i}")
        cap = temp_cap
        break
    temp_cap.release()

if cap is None or not cap.isOpened():
    print("❌ No webcam found.")
    exit()

# 📝 CSV setup
csv_file = "emotion_log.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Emotion"])

# ⏱ Detection setup
last_analysis_time = 0
analysis_interval = 2  # seconds
current_emotion = "Detecting..."
face_coords = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    current_time = time.time()
    if current_time - last_analysis_time > analysis_interval:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            current_emotion = result[0]['dominant_emotion'].capitalize()
            face_coords = result[0]['region']
            last_analysis_time = current_time

            # ⏱ Save to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, current_emotion])

            print(f"🧠 {timestamp} - {current_emotion}")

        except Exception as e:
            current_emotion = "No face"
            face_coords = None
            print("⚠️ Detection error:", e)

    # 🔲 Draw bounding box
    if face_coords:
        x, y, w, h = face_coords['x'], face_coords['y'], face_coords['w'], face_coords['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (203, 0, 143), 2)  # Pink color

    # 💗 Draw emotion text
    cv2.putText(frame, f'{current_emotion}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (203, 0, 143), 2, cv2.LINE_AA)

    # 🖼 Show camera
    cv2.imshow("🧠 Emotion Detection (Press 'q' or 'Esc' to quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC
        break

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed.")
