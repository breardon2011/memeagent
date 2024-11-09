import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import openai
import json
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# Initialize model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the existing FAISS index and ID map
index = faiss.read_index("meme_index.faiss")
with open("id_map.json", "r") as f:
    id_map = json.load(f)

def retrieve_similar_memes(query, top_k=5):
    # Convert text query to embedding
    inputs = processor(text=[query], return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    text_emb = text_emb.cpu().numpy()
    
    # Normalize and search in the FAISS index
    faiss.normalize_L2(text_emb)
    distances, indices = index.search(text_emb, top_k)
    
    # Retrieve image IDs and metadata of similar memes
    similar_memes = [(id_map[str(idx)], dist) for idx, dist in zip(indices[0], distances[0])]
    return similar_memes






def generate_caption_from_similar_memes(similar_memes):
    # Combine captions from similar memes
    context_captions = "\n".join([f"- {meme_id}" for meme_id, _ in similar_memes])

    # Prompt GPT-4 to create a new caption with similar themes
    prompt = f"Based on these similar meme captions:\n{context_captions}\nCreate a new, funny meme caption about {theme}."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    new_caption = response['choices'][0]['message']['content']
    return new_caption

# Generate a new caption based on similar memes
new_caption = generate_caption_from_similar_memes(similar_memes)
print("Generated caption:", new_caption)


def generate_image(theme):
    prompt = f"A funny image of a {theme}, perfect for a meme."
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    return image_url





def create_meme(image_url, caption):
    # Load the generated image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # Set up drawing context
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 30)  # Adjust font and size as needed
    
    # Define text position
    text_position = (10, img.height - 40)
    
    # Draw the caption onto the image
    draw.text(text_position, caption, (255, 255, 255), font=font)
    
    # Save and show the final meme
    img.show()
    img.save("generated_meme.jpg")



def generate_complete_meme(query, top_k=5):
    """
    Generates a complete meme by combining retrieval, caption generation, and image creation steps.
    
    Args:
        query (str): The theme or topic for the meme
        top_k (int): Number of similar memes to retrieve
    
    Returns:
        str: Path to the generated meme image
    """
    # Step 1: Retrieve similar memes
    similar_memes = retrieve_similar_memes(query, top_k)
    
    # Step 2: Generate new caption
    context_captions = "\n".join([f"- {meme_id}" for meme_id, _ in similar_memes])
    prompt = f"Based on these similar meme captions:\n{context_captions}\nCreate a new, funny meme caption about {query}."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    caption = response['choices'][0]['message']['content']
    
    # Step 3: Generate image
    image_url = generate_image(query)
    
    # Step 4: Create final meme
    create_meme(image_url, caption)
    
    return "generated_meme.jpg"