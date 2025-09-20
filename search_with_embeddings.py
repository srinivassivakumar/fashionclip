import torch
import open_clip

EMBEDDINGS_FILE = "D:/Sri/mymiroo/products/image_embeddings.pt"

# Load precomputed embeddings
print("Loading precomputed embeddings...")
data = torch.load(EMBEDDINGS_FILE)
image_features = data["features"]
image_files = data["files"]

# Load model + tokenizer (for text only now)
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Ask query
query = input("Enter your fashion query: ")

with torch.no_grad():
    text_features = model.encode_text(tokenizer([query]))
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Compare with precomputed image features
scores = (image_features @ text_features.T).squeeze().tolist()

# Rank results
ranked = sorted(zip(image_files, scores), key=lambda x: x[1], reverse=True)

print("\nTop matches:")
for path, score in ranked[:5]:
    print(f"{path} -> {score:.4f}")
