import os
import torch
import torch.nn.functional as F
import faiss
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import PeftModel
from tqdm import tqdm
from dataloader import create_dataloaders

class SearchEngine:
    """A class that encapsulates the entire retrieval pipeline, from embedding extraction to FAISS indexing and querying.
    Args:
        base_model_id: The HuggingFace model ID for the base SigLIP model (e.g., "google/siglip-base-patch16-224")
        lora_weights_dir: The directory where the LoRA adapter weights are stored (e.g., "./siglip_lora_model")
        device: The device to run the model on ("cuda" or "cpu")
    """

    def __init__(self, base_model_id: str = "google/siglip-base-patch16-224", 
                 lora_weights_dir: str = "./siglip_lora_model", device: str = "cuda"):

        self.device = device

        # Load the processor for both image and text inputs
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        
        # Load the base model
        print("Loading base model...")
        base_model = AutoModel.from_pretrained(base_model_id)
        
        # Inject LoRA adapters
        print("Injecting LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, lora_weights_dir)
        self.model.to(self.device)
        self.model.eval()

        # FAISS Index Initialization
        self.embedding_dim = self.model.config.text_config.hidden_size
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # A simple registry to map FAISS integer IDs to actual image paths
        self.image_registry = []

    @torch.no_grad()
    def build_index(self, dataloader: torch.utils.data.DataLoader):
        """Passes the image gallery through the Vision Encoder and adds them to FAISS.
        Args:
            dataloader: A PyTorch DataLoader that yields batches of images and their corresponding file paths.
        """

        print(f"Extracting visual embeddings for FAISS index...")
        
        for batch in tqdm(dataloader):
            # Move pixel values to the appropriate device and keep track of image paths for registry
            pixel_values = batch["pixel_values"].to(self.device)

            # Keep track of image paths and labels for the registry
            image_paths = batch["image_path"]
            label_names = batch["label_name"]
            
            # Extract Vision Features
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.pooler_output
            
            # L2 Normalize as a part of cosine similarity calculations
            image_embeds = F.normalize(image_embeds, dim=-1, p=2)
            
            # Convert to float32 numpy arrays for FAISS
            embeddings_np = image_embeds.cpu().numpy().astype('float32')
            
            # Add to FAISS and record the file path
            self.index.add(embeddings_np)
            for path, label in zip(image_paths, label_names):
                self.image_registry.append({"path": path, "label": label})

        print(f"Successfully indexed {self.index.ntotal} images into FAISS.")

    @torch.no_grad()
    def search(self, query: str, top_k: int = 3) -> list:
        """Passes text through the Text Encoder and queries FAISS.
        Args:
            query: The user's search query
            top_k: The number of top matches to return
        Returns:
            A list of dictionaries containing 'path' and 'similarity' for each match
        """

        # Process the text query
        inputs = self.processor(text=query, return_tensors="pt", padding="max_length", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        kwargs = {"input_ids": input_ids}

        if 'attention_mask' in inputs:
            attention_mask = inputs.attention_mask.to(self.device)
            kwargs['attention_mask'] = attention_mask
        
        # Extract text features
        text_outputs = self.model.text_model(**kwargs)
        text_embeds = text_outputs.pooler_output
        
        # L2 Normalize the text embedding
        text_embeds = F.normalize(text_embeds, dim=-1, p=2)
        query_np = text_embeds.cpu().numpy().astype('float32')
        
        # Search FAISS
        similarities, indices = self.index.search(query_np, top_k)
        
        # Map IDs back to file paths
        results = []
        for score, idx in zip(similarities[0], indices[0]):
            registry_entry = self.image_registry[idx]
            results.append({
                "path": registry_entry["path"],
                "label": registry_entry["label"],
                "similarity": score
            })
            
        return results

    def display_results(self, query: str, results: list):
        """Helper function to pop open a Matplotlib window with the top matches.
        Args:
            query: The original search query string (for display purposes)
            results: A list of dictionaries containing 'path' and 'similarity' for each match
        """

        # Create a grid of images with their similarity scores as titles
        fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
        fig.suptitle(f'Search Query: "{query}"', fontsize=16)
        
        # Handle case where top_k is 1
        if len(results) == 1:
            axes = [axes]

        # Display each result
        for ax, res in zip(axes, results):
            img = Image.open(res['path']).convert("RGB")
            ax.imshow(img)
            ax.axis('off')
            # Display the similarity score (0.0 to 1.0)
            ax.set_title(f"Score: {res['similarity']:.3f}\nTrue: {res['label']}", 
                         color='green', fontsize=10, wrap=True)
            
        plt.tight_layout()
        plt.show()

# --- Execution ---
if __name__ == "__main__":
    # Load the test data
    _, _, test_loader, _ = create_dataloaders(
        train_dir="./data/stanford_cars",
        test_dir="./data/stanford_cars",
        llm_captions_path="./data/llm_captions.json",
        processor_name="google/siglip-base-patch16-224",
        batch_size=64,
        num_workers=8,
        )
    
    # Initialize search engine
    engine = SearchEngine(base_model_id="google/siglip-base-patch16-224",
                          lora_weights_dir="./siglip_lora_model",
                          device="cuda")
    
    # Build the database
    engine.build_index(test_loader)
    
    # --- The Interactive CLI Loop ---
    print("\n" + "=" * 50)
    print("RETRIEVAL TERMINAL")
    print("=" * 50)
    print("Type a description of a vehicle to search in the database.")
    print("Type 'exit' or 'quit' to close the terminal.\n")
    
    while True:
        # Get user input
        query = input("Search Query > ")
        if query.lower() in ['exit', 'quit']:
            print("Shutting down terminal...")
            break
            
        if not query.strip():
            continue
            
        # Run the search
        results = engine.search(query=query, top_k=3)
        
        # Print results to terminal
        print("\n--- Top Matches ---")
        for i, res in enumerate(results):
            print(f"{i+1}. Score: {res['similarity']:.3f} | True Label: {res['label']:<30} | File: {os.path.basename(res['path'])}")
        print("-" * 19 + "\n")
        
        # Pop open the images visually
        engine.display_results(query, results)