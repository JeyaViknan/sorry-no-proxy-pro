import os
import cv2
import pickle
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1)  # CPU mode

DATASET_PATH = "data"
database = {}

print("Processing dataset...")

for file in os.listdir(DATASET_PATH):
    if file.endswith(".png") or file.endswith(".jpg"):
        path = os.path.join(DATASET_PATH, file)
        img = cv2.imread(path)

        faces = app.get(img)

        if len(faces) == 0:
            print(f"No face found in {file}")
            continue

        embedding = faces[0].embedding

        # Enrollment number = filename without extension
        student_id = os.path.splitext(file)[0]

        database[student_id] = embedding
        print(f"Processed: {student_id}")

# Save embeddings
with open("face_db.pkl", "wb") as f:
    pickle.dump(database, f)

print("\n✅ Embeddings saved as face_db.pkl")