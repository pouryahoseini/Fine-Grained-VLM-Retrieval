import os
import json
from typing import Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from evaluate import RetrievalEvaluator
from dataloader import create_dataloaders

# --- The Upgraded Training Loop ---
def train_model(train_dataset: torch.utils.data.Dataset, val_dataloader: torch.utils.data.DataLoader, 
                model_id: str = "google/siglip-base-patch16-224", batch_size: int = 16,
                num_workers: int = 8, epochs: int = 10, accum_steps: int = 4, patience: int = 3, 
                warmup_ratio: float = 0.1, lr: float = 5e-5, weight_decay: float = 0.01, 
                autocast_dtype: torch.dtype = torch.bfloat16,
                save_dir: str = "./siglip_lora_model", log_dir: str = "./siglip_lora_model/tensorboard_logs") -> torch.nn.Module:
    """Trains a LoRA adapter on top of a pre-trained model using Distributed Data Parallel (DDP) for multi-GPU training, 
    with support for mixed precision and gradient accumulation.
    Args:
        train_dataset: The training dataset to use for fine-tuning.
        val_dataloader: The validation dataloader for evaluating the model after each epoch.
        model_id: The identifier of the pre-trained model to load.
        batch_size: The number of samples per batch for training.
        num_workers: The number of subprocesses to use for data loading.
        epochs: The maximum number of epochs to train for.
        accum_steps: The number of steps to accumulate gradients before performing an optimizer step.
        patience: The number of epochs to wait for improvement before early stopping.
        warmup_ratio: The fraction of total training steps to use for learning rate warmup.
        lr: The initial learning rate for the optimizer.
        weight_decay: The weight decay coefficient for regularization.
        autocast_dtype: The data type to use for automatic mixed precision (e.g., torch.float16 or torch.bfloat16).
        save_dir: The directory where the best model will be saved.
        log_dir: The directory where TensorBoard logs will be saved.
    Returns:
        The best model after training, loaded into memory.
    """
    
    # Setup Distributed Data Parallel (DDP) environment
    local_rank, global_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = global_rank == 0

    # TensorBoard Setup (Only log on the main process to avoid corrupted files)
    if is_main_process:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[{global_rank}] TensorBoard initialized.")

    # DDP DataLoader Setup
    # The DistributedSampler ensures each GPU gets a unique chunk of the dataset
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=num_workers
    )

    # Model & LoRA Setup
    model = setup_peft_model(model_id).to(device)
    
    # Wrap model in DDP if running distributed
    if world_size > 1:
        # peft models usually require find_unused_parameters=True
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Optimizer Setup with parameter groups
    # Standard practice is to not apply weight decay to bias and LayerNorm parameters
    decay_parameters = []
    no_decay_parameters = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        # Heuristic: weight matrices are > 1D, biases and norms are 1D or 0D
        if param.ndim > 1:
            decay_parameters.append(param)
        else:
            no_decay_parameters.append(param)

    optimizer_grouped_parameters = [
        {'params': decay_parameters, 'weight_decay': weight_decay},
        {'params': no_decay_parameters, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    
    # AMP setup
    # GradScaler is technically only required for float16
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16))
    
    # Scheduler Setup
    total_optimization_steps = (len(train_loader) // accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_optimization_steps * warmup_ratio), num_training_steps=total_optimization_steps
    )

    # Set initial global step and best validation metric for early stopping
    global_step = 0
    best_val_recall = 0.0
    patience_counter = 0

    # --- Epoch Loop ---
    for epoch in range(epochs):
        print(f"\n**********\nEpoch {epoch+1} of {epochs} | Global Step: {global_step} | World Size: {world_size}")
        
        # Set the epoch on the sampler so it shuffles data differently each epoch
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # --- Training Loop ---
        model.train()
        for step, batch in enumerate(train_loader):
            
            # --- Automatic Mixed Precision (AMP) Context ---
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):

                # Forward pass through the model
                kwargs = {
                    "pixel_values": batch["pixel_values"].to(device),
                    "input_ids": batch["input_ids"].to(device),
                }

                if "attention_mask" in batch:
                    kwargs["attention_mask"] = batch["attention_mask"].to(device)
                
                outputs = model(**kwargs)
                
                # If wrapped in DDP, access base model parameters via model.module
                base_model = model.module if isinstance(model, DDP) else model
                
                # Compute the pairwise sigmoid loss
                loss = pairwise_sigmoid_loss(
                    image_embeds=outputs.image_embeds,
                    text_embeds=outputs.text_embeds,
                    logit_scale=base_model.logit_scale,
                    logit_bias=base_model.logit_bias
                )
                
                # Scale loss by accumulation steps
                loss = loss / accum_steps

            # Backward pass with Scaler
            scaler.scale(loss).backward()

            # --- Gradient Accumulation Step ---
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                # Unscale before clipping
                scaler.unscale_(optimizer)

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step the optimizer and scheduler, then zero gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # slightly faster than zero_grad()
                scheduler.step()

                # Increment global step for logging
                global_step += 1

                # --- TensorBoard Logging ---
                if is_main_process and global_step % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    # Multiply by accum_steps to log the true loss value, although it is only for one physical step
                    writer.add_scalar("Train/Loss", loss.item() * accum_steps, global_step)
                    writer.add_scalar("Train/LearningRate", current_lr, global_step)
                    writer.add_scalar("Model/Bias", base_model.logit_bias.item(), global_step)
                    writer.add_scalar("Model/Temperature_Scale", base_model.logit_scale.item(), global_step)

        # --- Validation & Early Stopping ---
        stop_training = torch.tensor(0, device=device)
        
        if is_main_process:
            # Run validation and generate report
            print(f"Epoch {epoch+1} complete. Running evaluation...")
            base_model = model.module if isinstance(model, DDP) else model
            evaluator = RetrievalEvaluator(model=base_model, dataloader=val_dataloader, device=device)
            metrics = evaluator.generate_report()
            
            # Log validation metrics to TensorBoard
            current_val_recall = metrics['Recall@1']
            writer.add_scalar("Val/Recall@1", current_val_recall, epoch)
            writer.add_scalar("Val/mAP", metrics['mAP'], epoch)
            
            # Early stopping logic based on Recall@1
            if current_val_recall > best_val_recall:
                best_val_recall = current_val_recall
                patience_counter = 0
                print(f"New Best Recall@1: {best_val_recall * 100:.2f}%. Saving model...")
                os.makedirs(save_dir, exist_ok=True)
                base_model.save_pretrained(save_dir)
            else:
                patience_counter += 1
                print(f"Recall did not improve. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("\nEarly stopping triggered! Model convergence detected.")
                    stop_training += 1

        # Synchronize early stopping flag across all processes
        if world_size > 1:
            dist.all_reduce(stop_training, op=dist.ReduceOp.SUM)
        
        # If any process has triggered early stopping, break out of the epoch loop
        if stop_training.item() > 0:
            break

    # Cleanup
    if is_main_process:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()
    
    return base_model.module if isinstance(model, DDP) else base_model

def setup_ddp() -> Tuple[int, int, int]:
    """Initializes the distributed backend."""

    # Get environment variables that are automatically injected by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Only initialize distributed training if we are actually running distributed
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        
    return local_rank, global_rank, world_size

def setup_peft_model(model_id: str = "google/siglip-base-patch16-224") -> torch.nn.Module:
    """Loads the base model and sets up the LoRA configuration for fine-tuning.
    Args:
        model_id: The identifier of the pre-trained model to load.
    Returns:
        The PEFT-wrapped model ready for training.
    """

    print(f"Loading base foundation model: {model_id}")
    model = AutoModel.from_pretrained(model_id)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        # modules_to_save=["logit_scale", "logit_bias"] 
    )

    peft_model = get_peft_model(model, config)

    # Manually unfreeze logit_scale and logit_bias,
    # because peft's module_to_save doesn't actually unfreeze them as they are parameters, not modules.
    for name, param in peft_model.named_parameters():
        if "logit_scale" in name or "logit_bias" in name:
            param.requires_grad = True

    # Print trainable parameters for verification
    peft_model.print_trainable_parameters()

    return peft_model.to("cuda")

def pairwise_sigmoid_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, 
                          logit_scale: torch.Tensor, logit_bias: torch.Tensor) -> torch.Tensor:
    """Computes a pairwise sigmoid loss that encourages matching image-text pairs to have higher similarity than non-matching pairs.
    Args:
        image_embeds: Tensor of shape (batch_size, embed_dim) containing image embeddings.
        text_embeds: Tensor of shape (batch_size, embed_dim) containing text embeddings.
        logit_scale: Tensor of shape (1,) containing the logit scale parameter.
        logit_bias: Tensor of shape (1,) containing the logit bias parameter.
    Returns:
        The computed pairwise sigmoid loss.
    """

    # Normalize embeddings to unit length and compute scaled and shifted cosine similarity
    image_embeds = F.normalize(image_embeds, dim=-1, p=2)
    text_embeds = F.normalize(text_embeds, dim=-1, p=2)
    logits = torch.matmul(image_embeds, text_embeds.t()) * logit_scale.exp() + logit_bias
    
    # Compute the loss using the pairwise sigmoid formulation
    N = logits.size(0)
    labels = 2 * torch.eye(N, device=logits.device) - 1
    loss = -F.logsigmoid(labels * logits).sum() / N

    return loss

# --- Main Execution ---
if __name__ == "__main__":

    # Define Paths and Parameters
    data_dir = "./data/stanford_cars"
    llm_captions_path = "./data/llm_captions.json"
    processor_name = "google/siglip-base-patch16-224"
    model_id = "google/siglip-base-patch16-224"
    batch_size = 16
    accum_steps = 4
    epochs = 10
    patience = 3
    warmup_ratio = 0.1
    lr = 5e-5
    weight_decay = 0.02
    num_workers = 8
    autocast_dtype = torch.bfloat16
    save_dir = "./siglip_lora_model"
    log_dir = "./siglip_lora_model/tensorboard_logs"

    # Setup hardware device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dataloaders with internal validation split
    train_loader, val_loader, test_loader, processor = create_dataloaders(
        train_dir=data_dir,
        test_dir=data_dir,
        llm_captions_path=llm_captions_path,
        processor_name=processor_name,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=0.1
        )
    
    # Launch Training
    trained_model = train_model(train_loader.dataset, val_loader, model_id=model_id, batch_size=batch_size,
                                num_workers=num_workers, epochs=epochs, accum_steps=accum_steps, 
                                patience=patience, warmup_ratio=warmup_ratio, lr=lr, weight_decay=weight_decay, 
                                autocast_dtype=autocast_dtype, save_dir=save_dir, log_dir=log_dir)
    trained_model.eval()
    
    # Load the original pre-trained model (before fine-tuning) to establish the zero-shot baseline
    print(f"Loading base model: {model_id}...")
    base_model = AutoModel.from_pretrained(model_id).to(device)
    base_model.eval()

    # Run evaluation on the test set using the trained and base models to compare performance
    print(f"\nTRAINED MODEL EVALUATION:")
    trained_evaluator = RetrievalEvaluator(trained_model, test_loader, device=device)
    trained_metrics = trained_evaluator.generate_report()
    
    print(f"\nBASE MODEL EVALUATION:")
    base_evaluator = RetrievalEvaluator(base_model, test_loader, device=device)
    baseline_metrics = base_evaluator.generate_report()

    # Save the evaluation metrics to a JSON file for later analysis
    evaluation_results = {
        "trained_model": trained_metrics,
        "baseline_model": baseline_metrics
    }
    with open(os.path.join(save_dir, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"\nEvaluation complete! Results saved to {os.path.join(save_dir, 'evaluation_results.json')}")