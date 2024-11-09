import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import faiss  # or use an alternative vector database
import numpy as np
import json


# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


metadata_path = 'archive/posts.csv'  # Update with the path to the CSV file
images_path = 'archive/images/'  # Update with the path to the images folder
metadata = pd.read_csv(metadata_path)

# Prepare lists to store embeddings and metadata
image_embeddings = []
image_ids = []  # You can store IDs or filenames to retrieve images later

for index, row in metadata.iterrows():
    try:
        # Load image and preprocess it
        image_path = os.path.join(images_path, f"{row['id']}.png")
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess and move to device
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Get the image embeddings
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs)
        
        # Normalize and store embedding
        image_embeddings.append(image_emb.cpu().numpy())
        image_ids.append(row['id'])
        
    except Exception as e:
        print(f"Error processing {row['id']}: {e}")
        continue

# Convert list to a 2D NumPy array for compatibility with FAISS

image_embeddings = np.vstack(image_embeddings)


# Initialize a FAISS index for cosine similarity
d = image_embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
faiss.normalize_L2(image_embeddings)  # Normalize embeddings for cosine similarity

# Add image embeddings to the index
index.add(image_embeddings)

# Save image metadata mapping (ID or filename) for later use
id_map = {i: image_id for i, image_id in enumerate(image_ids)}

# Save FAISS index
faiss.write_index(index, "meme_index.faiss")

# Save metadata as JSON or pickle
with open("id_map.json", "w") as f:
    json.dump(id_map, f)


def search_images_by_text(query, top_k=5):
    # Process the text query
    inputs = processor(text=[query], return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    text_emb = text_emb.cpu().numpy()
    
    # Normalize text embedding and search
    faiss.normalize_L2(text_emb)
    distances, indices = index.search(text_emb, top_k)
    
    # Retrieve image filenames and distances
    results = [(id_map[idx], dist) for idx, dist in zip(indices[0], distances[0])]
    return results

# Example usage:
results = search_images_by_text("funny cat meme")
print("Top matches:", results)


