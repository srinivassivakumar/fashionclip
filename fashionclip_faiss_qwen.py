import os
import torch
import open_clip
import faiss
import requests

OLLAMA_API = "http://localhost:11434/api/chat"
EMBEDDINGS_FILE = "D:/Sri/mymiroo/S-AI/fashionclip/products/image_embeddings.pt"
FAISS_INDEX_FILE = "D:/Sri/mymiroo/S-AI/fashionclip/products/image_index.faiss"
FILES_LIST_FILE = "D:/Sri/mymiroo/S-AI/fashionclip/products/image_files.pt"

print("🚀 Starting pipeline...")

# === Step 1: Build FAISS index if missing ===
if not (os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FILES_LIST_FILE)):
    print("🔧 Building FAISS index...")
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"❌ Missing embeddings file: {EMBEDDINGS_FILE}")

    data = torch.load(EMBEDDINGS_FILE)
    image_features = data["features"].cpu().numpy()
    image_files = data["files"]

    dim = image_features.shape[1]
    print(f"📐 Feature dimension = {dim}, Images = {len(image_files)}")

    index = faiss.IndexFlatIP(dim)
    index.add(image_features)

    faiss.write_index(index, FAISS_INDEX_FILE)
    torch.save(image_files, FILES_LIST_FILE)
    print("✅ FAISS index built and saved.")
else:
    print("📂 FAISS index already exists.")

# === Step 2: Load FAISS + files ===
print("📥 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)
image_files = torch.load(FILES_LIST_FILE)
print(f"✅ Loaded index with {len(image_files)} images.")

# === Step 3: Load OpenCLIP ===
print("🧠 Loading OpenCLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# === Step 4: Query ===
query = input("\nEnter your fashion query: ")
print(f"🔎 Encoding query: {query}")

with torch.no_grad():
    text_features = model.encode_text(tokenizer([query]))
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()

print("✅ Query encoded. Shape:", text_features.shape)

# === Step 5: Search FAISS ===
print("🔍 Searching FAISS...")
k = 3
scores, indices = index.search(text_features, k)
print("✅ Search complete.")

top_images = [image_files[i] for i in indices[0]]
print("\n🎯 Top matches:")
for img, score in zip(top_images, scores[0]):
    print(f" - {img} (score={score:.4f})")

# === Step 6: Ask Qwen ===
prompt = f"""
I searched for: "{query}".
Here are 3 product image results:
{top_images}

Can you describe the style of these products
and suggest how to combine them in an outfit?
"""

print("\n🤖 Calling Qwen...")
resp = requests.post(
    OLLAMA_API,
    json={"model": "qwen2.5:3b",
          "messages": [{"role": "user", "content": prompt}],
          "stream": False},
    timeout=300
)

print("✅ Ollama API called. Status:", resp.status_code)

try:
    data = resp.json()
    print("\n--- Raw Response ---")
    print(data)
    print("\n💬 Qwen says:\n", data["message"]["content"])
except Exception as e:
    print("⚠️ Failed to parse Qwen response:", e)
