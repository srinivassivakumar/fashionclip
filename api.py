import torch, faiss, open_clip, requests
from fastapi import FastAPI
from pydantic import BaseModel

# --- Config ---
OLLAMA_API = "http://localhost:11434/api/chat"
EMBEDDINGS_FILE = "D:/Sri/mymiroo/S-AI/fashionclip/products/image_embeddings.pt"
FAISS_INDEX_FILE = "D:/Sri/mymiroo/S-AI/fashionclip/products/image_index.faiss"
FILES_LIST_FILE = "D:/Sri/mymiroo/S-AI/fashionclip/products/image_files.pt"

# Load FAISS + OpenCLIP once at startup
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)
image_files = torch.load(FILES_LIST_FILE)

print("Loading OpenCLIP...")
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# --- API app ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search")
def search(req: QueryRequest):
    # Encode text
    with torch.no_grad():
        text = tokenizer([req.query])
        text = {k: v.to(device) for k, v in text.items()} if isinstance(text, dict) else text.to(device)
        text_feat = model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        q = text_feat.cpu().numpy()

    # FAISS search
    scores, idx = index.search(q, req.top_k)
    top_imgs = [image_files[i] for i in idx[0]]
    scored_results = [{"image": f, "score": float(s)} for f, s in zip(top_imgs, scores[0])]

    # Ask Qwen
    prompt = f"""
    User query: "{req.query}"
    Here are the top {req.top_k} product matches: {top_imgs}

    Describe the style of these products and suggest how to combine them in an outfit.
    """

    resp = requests.post(OLLAMA_API, json={
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })

    try:
        qwen_reply = resp.json()["message"]["content"]
    except Exception:
        qwen_reply = "No reply from Qwen."

    return {"results": scored_results, "qwen_advice": qwen_reply}
