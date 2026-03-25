import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# ----------------------------
# Load saved embeddings
# ----------------------------
with open("face_db.pkl", "rb") as f:
    database = pickle.load(f)

# ----------------------------
# Initialize InsightFace
# ----------------------------
app = FaceAnalysis()
app.prepare(ctx_id=-1)  # CPU mode

# ----------------------------
# Cosine similarity function
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# Test image path
# ----------------------------
TEST_IMAGE = "data/23BCE1766.png"   # 👈 change this if needed

# ----------------------------
# Read image
# ----------------------------
img = cv2.imread(TEST_IMAGE)

if img is None:
    print("❌ Image not found. Check path.")
    exit()

# ----------------------------
# Detect face
# ----------------------------
faces = app.get(img)

if len(faces) == 0:
    print("❌ No face detected")
    exit()

# ----------------------------
# Extract embedding
# ----------------------------
emb = faces[0].embedding

# ----------------------------
# Compare with database
# ----------------------------
best_match = "Unknown"
best_score = 0
second_best = 0

for student_id, db_emb in database.items():
    score = cosine_similarity(db_emb, emb)

    if score > best_score:
        second_best = best_score
        best_score = score
        best_match = student_id

# ----------------------------
# Apply threshold + safety margin
# ----------------------------
if best_score > 0.5 and (best_score - second_best) > 0.1:
    print(f"✅ Match: {best_match} (score: {best_score:.2f})")
else:
    print(f"❌ Unknown (score: {best_score:.2f})")

# ----------------------------
# (Optional) Show image
# ----------------------------
box = faces[0].bbox.astype(int)
cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)

cv2.putText(img, best_match, (box[0], box[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()