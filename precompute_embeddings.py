import os
import torch
from PIL import Image
import open_clip

# Path to your product images
IMAGE_FOLDER = "D:/Sri/mymiroo/S-AI/fashionclip/products"
EMBEDDINGS_FILE = "D:/Sri/mymiroo/S-AI/fashionclip/products/image_embeddings.pt"

# Load OpenCLIP model
print("Loading OpenCLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Collect images
image_files = [
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".png"))
]

if not image_files:
    raise ValueError(f"No images found in {IMAGE_FOLDER}")

print(f"Found {len(image_files)} images. Precomputing embeddings...")

# Preprocess + encode all images
images = [preprocess(Image.open(path).convert("RGB")).unsqueeze(0) for path in image_files]
images = torch.cat(images)

with torch.no_grad():
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

# Save embeddings + file mapping
torch.save({"features": image_features, "files": image_files}, EMBEDDINGS_FILE)
print(f"✅ Saved embeddings to {EMBEDDINGS_FILE}")
