import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# ----------------------------
# Load database
# ----------------------------
with open("face_db.pkl", "rb") as f:
    database = pickle.load(f)

# ----------------------------
# Initialize model
# ----------------------------
app = FaceAnalysis()
app.prepare(ctx_id=-1)

# ----------------------------
# Cosine similarity
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# Start webcam
# ----------------------------
cap = cv2.VideoCapture(0)

print("🎥 Camera started... Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error")
        break

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding

        best_match = "Unknown"
        best_score = 0
        second_best = 0

        # Compare with all students
        for student_id, db_emb in database.items():
            score = cosine_similarity(db_emb, emb)

            if score > best_score:
                second_best = best_score
                best_score = score
                best_match = student_id

        # 🔥 Decision logic
        if best_score > 0.5 and (best_score - second_best) > 0.1:
            name = best_match
            color = (0, 255, 0)   # Green
        else:
            name = "Unknown"
            color = (0, 0, 255)   # Red

        # Draw bounding box
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Show name + confidence
        cv2.putText(frame, f"{name} ({best_score:.2f})",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    cv2.imshow("🎓 Attendance Face Recognition", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()