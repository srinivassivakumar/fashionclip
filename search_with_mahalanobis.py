''''
Cosine and Euclidean similarity treat all features equally, meaning every detail about an image—like color,
 shape, or texture—counts the same, even if some details are noisy or overlapping. Mahalanobis is smarter 
 because it looks at the whole dataset to figure out which features usually move together and which ones are 
 just random noise, then reduces the importance of those. On small datasets, like 200 images, it doesn’t 
 have enough information to learn these patterns, so it behaves almost the same as cosine or Euclidean. 
 But on large datasets, with thousands of images, it can spot real correlations
(like color and brightness always moving together) and ignore noisy features, which makes its similarity 
scores closer to how humans would judge similarity.

'''


import torch
import open_clip
import numpy as np
from numpy.linalg import inv

# Path to precomputed embeddings
EMBEDDINGS_FILE = "D:/Sri/mymiroo/products/image_embeddings.pt"

# -------------------------------
# Load precomputed embeddings
# -------------------------------
print("Loading precomputed embeddings...")
data = torch.load(EMBEDDINGS_FILE)
image_features = data["features"].cpu().numpy()   # embeddings as numpy array
image_files = data["files"]

# -------------------------------
# Compute covariance + inverse
# -------------------------------
print("Computing covariance matrix (may take a moment)...")
cov_matrix = np.cov(image_features, rowvar=False)
inv_cov_matrix = inv(cov_matrix)

# -------------------------------
# Load model + tokenizer
# -------------------------------
print("Loading OpenCLIP model...")
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# -------------------------------
# Get user query
# -------------------------------
query = input("Enter your fashion query: ")

with torch.no_grad():
    text_features = model.encode_text(tokenizer([query]))
text_features = text_features.cpu().numpy().reshape(1, -1)  # ensure 2D

# -------------------------------
# Mahalanobis distance function
# -------------------------------
def mahalanobis(u, v, inv_cov):
    diff = u - v
    dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    return float(dist)  # convert to scalar float

# -------------------------------
# Compare query with all images
# -------------------------------
scores = []
for idx, img_vec in enumerate(image_features):
    dist = mahalanobis(text_features, img_vec.reshape(1, -1), inv_cov_matrix)
    scores.append((image_files[idx], dist))

# Rank by *lowest* distance (smaller = closer match)
ranked = sorted(scores, key=lambda x: x[1])

# -------------------------------
# Show top matches
# -------------------------------
print("\nTop matches (Mahalanobis distance):")
for path, dist in ranked[:5]:
    print(f"{path} -> distance: {dist:.4f}")
