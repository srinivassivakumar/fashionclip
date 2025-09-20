# search_server.py
import torch
import open_clip
import os


# Load model once
print("Loading OpenCLIP...")
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Load precomputed embeddings
data = torch.load("D:/Sri/mymiroo/S-AI/fashionclip/products/image_embeddings.pt")
image_features = data["features"]
image_files = data["files"]

print("✅ Ready! Type your query below (or 'quit' to exit).")

while True:
    query = input("\nEnter query: ")
    if query.lower() in ["quit", "exit"]:
        break

    with torch.no_grad():
        text_features = model.encode_text(tokenizer([query]))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    scores = (image_features @ text_features.T).squeeze().tolist()
    ranked = sorted(zip(image_files, scores), key=lambda x: x[1], reverse=True)

    for path, score in ranked[:5]:
        print(f"{path} -> {score:.4f}")
