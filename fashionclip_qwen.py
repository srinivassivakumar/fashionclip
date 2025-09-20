import os
import torch
import open_clip
import requests
from PIL import Image

# --- Settings ---
OLLAMA_API = "http://localhost:11434/api/chat"
EMBEDDINGS_FILE = "D:/Sri/mymiroo/products/image_embeddings.pt"

# === Step 1: Load precomputed FashionCLIP embeddings ===
print("Loading precomputed FashionCLIP embeddings...")
data = torch.load(EMBEDDINGS_FILE)
image_features = data["features"]
image_files = data["files"]
print(f"Loaded {len(image_files)} images with embeddings.")

# === Step 2: Load OpenCLIP for query encoding ===
print("Loading OpenCLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# === Step 3: Take user query & encode ===
query = input("\nEnter your fashion query: ")
print(f"Encoding query: {query}")

with torch.no_grad():
    text_features = model.encode_text(tokenizer([query]))
    text_features /= text_features.norm(dim=-1, keepdim=True)

# === Step 4: Rank images by similarity ===
print("Ranking images...")
scores = (image_features @ text_features.T).squeeze().tolist()
ranked = sorted(zip(image_files, scores), key=lambda x: x[1], reverse=True)
top_images = [path for path, _ in ranked[:3]]

print("\nTop matches (from FashionCLIP):")
for img in top_images:
    print(f" - {img}")

# === Step 5: Ask Qwen for reasoning ===
prompt = f"""
I searched for: "{query}".
Here are 3 product image results:
{top_images}

Can you describe the style of these products
and suggest how to combine them in an outfit?
"""

print("\nCalling Qwen via Ollama API...")
resp = requests.post(
    OLLAMA_API,
    json={
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    },
    timeout=300
)

print("\n--- Ollama API Response ---")
print(resp.status_code)
print(resp.json())

# === Step 6: Show just Qwen's answer ===
try:
    answer = resp.json()["message"]["content"]
    print("\nQwen says:\n", answer)
except Exception as e:
    print("⚠️ Could not extract Qwen's answer:", e)
