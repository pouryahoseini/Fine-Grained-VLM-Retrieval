import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel
from typing import Tuple
from dataloader import create_dataloaders

class RetrievalEvaluator:
    """Evaluates the performance of the trained model on the test set using standard retrieval metrics.
    Metrics include:
    - Recall@K (for K=1, 5, 10)
    - Mean Reciprocal Rank (MRR)
    - Mean Average Precision (mAP)
    The evaluator extracts embeddings for all test images and their corresponding text captions, computes a similarity matrix, 
    and then calculates the metrics based on the ranks of the correct matches.
    Args:
        model: The trained model to evaluate.
        dataloader: The dataloader for the test set.
        device: The device to run the evaluation on.
    Returns:
        A dictionary containing the calculated metrics.
    """

    def __init__(self, model: object, dataloader: object, device: str = "cuda"):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    @torch.no_grad()
    def extract_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts and normalizes all embeddings from the dataset.
        Returns:
            image_embeds: Tensor of shape (num_samples, embed_dim) containing L2-normalized image embeddings.
            text_embeds: Tensor of shape (num_samples, embed_dim) containing L2-normalized text embeddings.
            labels: Tensor of shape (num_samples,) containing the class labels for each sample.
        """

        # Set the model to evaluation mode
        self.model.eval()
        
        # Initialize lists to store embeddings and labels
        all_image_embeds = []
        all_text_embeds = []
        all_labels = []

        for batch in tqdm(self.dataloader):
            # Move tensors to the specified device
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(self.device)
            
            # We need the class ID to know which images map to which texts
            labels = batch["class_id"].to(self.device) 

            # Forward pass
            kwargs = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
            }

            if "attention_mask" in batch:
                kwargs["attention_mask"] = attention_mask
            
            outputs = self.model(**kwargs)

            # L2 Normalize the embeddings before saving them
            img_embeds = F.normalize(outputs.image_embeds, dim=-1, p=2)
            txt_embeds = F.normalize(outputs.text_embeds, dim=-1, p=2)

            # Append to the lists
            all_image_embeds.append(img_embeds)
            all_text_embeds.append(txt_embeds)
            all_labels.append(labels)

        return (
            torch.cat(all_image_embeds, dim=0),
            torch.cat(all_text_embeds, dim=0),
            torch.cat(all_labels, dim=0)
        )

    def generate_report(self) -> dict:
        """Runs the full evaluation suite and prints a formatted report.
        Returns:
            A dictionary containing the calculated metrics.
        """

        # Extract and normalize all embeddings from the test set
        print("Extracting embeddings...")
        image_embeds, text_embeds, labels = self.extract_embeddings()
        
        # Compute the cosine similarity matrix
        print("\nComputing similarity matrix...")
        similarity_matrix = torch.matmul(text_embeds, image_embeds.t())

        # Calculate metrics
        print("Calculating metrics...")
        recalls = calculate_recall_at_k(similarity_matrix, labels, labels, k_values=[1, 5, 10])
        mrr = calculate_mrr(similarity_matrix, labels, labels)
        m_ap = calculate_map(similarity_matrix, labels, labels)

        # Print the formal report
        print("-" * 40)
        print("RETRIEVAL METRICS REPORT")
        print("-" * 40)
        print(f"Recall@1:  {recalls['Recall@1'] * 100:>5.2f}%")
        print(f"Recall@5:  {recalls['Recall@5'] * 100:>5.2f}%")
        print(f"Recall@10: {recalls['Recall@10'] * 100:>5.2f}%")
        print(f"mAP:       {m_ap * 100:>5.2f}%")
        print(f"MRR:       {mrr:>5.4f}")
        print("-" * 40)
        
        return {**recalls, "MRR": mrr, "mAP": m_ap}

def calculate_recall_at_k(similarity_matrix: torch.Tensor, query_labels: torch.Tensor, 
                          gallery_labels: torch.Tensor, k_values: list = [1, 5, 10]) -> dict:
    """
    Calculates Recall@K. 
    Recall@K is 1 if ANY image of the correct class appears in the top K retrieved images.
    Args:
        similarity_matrix: Tensor of shape (num_queries, num_gallery) containing cosine similarities.
        query_labels: Tensor of shape (num_queries,) containing the class labels for the query set.
        gallery_labels: Tensor of shape (num_gallery,) containing the class labels for the gallery set.
        k_values: List of integers representing the K values for Recall@K.
    Returns:
        A dictionary containing Recall@K for each specified K.
    """

    # Sort the gallery for each query based on similarity (descending)
    _, top_indices = similarity_matrix.topk(max(k_values), dim=1, largest=True, sorted=True)
    
    # Map indices to actual class labels
    retrieved_labels = gallery_labels[top_indices]
    
    # Check if the retrieved labels match the query labels
    # Expand query_labels to match the shape: [num_queries, K]
    matches = retrieved_labels == query_labels.unsqueeze(1)
    
    # Calculate Recall@K for each K
    recalls = {}
    for k in k_values:
        # If there is at least one match in the top K for a query, it's a success (1.0)
        successes = matches[:, :k].any(dim=1).float()
        recalls[f"Recall@{k}"] = successes.mean().item()
        
    return recalls

def calculate_mrr(similarity_matrix: torch.Tensor, query_labels: torch.Tensor, gallery_labels: torch.Tensor) -> float:
    """
    Mean Reciprocal Rank (MRR).
    Looks at the rank of the first correct image. MRR = 1/rank.
    Args:
        similarity_matrix: Tensor of shape (num_queries, num_gallery) containing cosine similarities.
        query_labels: Tensor of shape (num_queries,) containing the class labels for the query set.
        gallery_labels: Tensor of shape (num_gallery,) containing the class labels for the gallery set.
    Returns:
        The Mean Reciprocal Rank (MRR) as a float.
    """

    # Sort all gallery images for each query
    _, sorted_indices = similarity_matrix.sort(dim=1, descending=True)
    retrieved_labels = gallery_labels[sorted_indices]
    matches = retrieved_labels == query_labels.unsqueeze(1)
    
    # Find the index of the first match (argmax returns the first True)
    # Adding a check in case a query has NO matches (rare, but mathematically possible)
    has_match = matches.any(dim=1)
    first_match_idx = matches.int().argmax(dim=1).float()
    
    # Calculate reciprocal rank (adding 1 because indices are 0-based)
    reciprocal_ranks = 1.0 / (first_match_idx + 1.0)
    
    # Zero out ranks for queries that had absolutely no matches
    reciprocal_ranks = reciprocal_ranks * has_match.float()
    
    return reciprocal_ranks.mean().item()

def calculate_map(similarity_matrix: torch.Tensor, query_labels: torch.Tensor, gallery_labels: torch.Tensor) -> float:
    """
    Mean Average Precision (mAP).
    Evaluates if ALL relevant images are ranked highly, not just the first one.
    Args:
        similarity_matrix: Tensor of shape (num_queries, num_gallery) containing cosine similarities.
        query_labels: Tensor of shape (num_queries,) containing the class labels for the query set.
        gallery_labels: Tensor of shape (num_gallery,) containing the class labels for the gallery set.
    Returns:
        The Mean Average Precision (mAP) as a float.
    """

    # Sort all gallery images for each query
    _, sorted_indices = similarity_matrix.sort(dim=1, descending=True)
    retrieved_labels = gallery_labels[sorted_indices]
    matches = (retrieved_labels == query_labels.unsqueeze(1)).float()
    
    # Cumulative sum of matches gives the number of true positives at each rank K
    true_positives_at_k = matches.cumsum(dim=1)
    
    # Precision at K: (True Positives at K) / K
    ranks = torch.arange(1, matches.size(1) + 1, device=matches.device).float().unsqueeze(0)
    precision_at_k = true_positives_at_k / ranks
    
    # We only care about precision at the exact ranks where a relevant image was retrieved
    average_precision = (precision_at_k * matches).sum(dim=1) / matches.sum(dim=1).clamp(min=1)
    
    return average_precision.mean().item()

if __name__ == "__main__":
    # Create dataloaders and processor
    _, _, test_loader, processor = create_dataloaders(
        train_dir="./data/stanford_cars",
        test_dir="./data/stanford_cars",
        llm_captions_path="./data/llm_captions.json",
        batch_size=200,
        num_workers=8
    )

    # Define hardware device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/siglip-base-patch16-224"

    # Load the original pre-trained model (before fine-tuning) to establish the zero-shot baseline
    print(f"Loading base model: {model_id}...")
    base_model = AutoModel.from_pretrained(model_id).to(device)
    base_model.eval()

    # Run evaluation on the test set using the base model to establish the zero-shot baseline
    print("\nStarting Zero-Shot Evaluation Pipeline...")
    evaluator = RetrievalEvaluator(base_model, test_loader, device=device)
    baseline_metrics = evaluator.generate_report()