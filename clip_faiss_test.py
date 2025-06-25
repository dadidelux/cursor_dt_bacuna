import os
import numpy as np
import torch
import clip
from PIL import Image
import faiss

# 1. Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Gather all image paths
image_dir = 'intercropping_classification/train'
image_paths = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images.")

# 3. Extract embeddings
embeddings = []
for path in image_paths:
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize
        embeddings.append(emb.cpu().numpy())
embeddings = np.vstack(embeddings).astype('float32')

# 4. Build FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # Cosine similarity (since vectors are normalized)
index.add(embeddings)

# 5. Save index and paths for later use
faiss.write_index(index, "intercropping_clip.index")
np.save("intercropping_image_paths.npy", np.array(image_paths))

print("Index and image paths saved.")

# 6. Test: Find similar images for a query
def get_clip_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype('float32')

def find_similar_images(query_image_path, top_k=5):
    query_emb = get_clip_embedding(query_image_path)
    D, I = index.search(query_emb, top_k)
    similar_paths = [image_paths[i] for i in I[0]]
    return similar_paths, D[0]

# Example usage:
query_path = image_paths[0]  # Use the first image as a query for testing
similar_images, scores = find_similar_images(query_path)
print('Query image:', query_path)
print('Most similar images:')
for path, score in zip(similar_images, scores):
    print(f"{path} (score: {score:.4f})")