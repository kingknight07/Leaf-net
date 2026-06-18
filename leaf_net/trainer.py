import torch
import time
from typing import Tuple, List

def train_model(model: torch.nn.Module, device: str, train_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, epochs: int, 
                print_freq: int = 10) -> List[float]:
    """
    Trains the EcoConvNet model.
    
    Args:
        model: The PyTorch model to train.
        device: 'cuda' or 'cpu'.
        train_loader: DataLoader for the training dataset.
        optimizer: The optimizer (e.g., Adam).
        criterion: The loss function (e.g., CrossEntropyLoss).
        epochs: Number of epochs to train.
        print_freq: Frequency of printing training progress.
        
    Returns:
        A list of average training losses per epoch.
    """
    model.to(device)
    model.train()
    print(f"--- Starting Training on {device} ---")
    
    history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        
        if (epoch + 1) % print_freq == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
    print("--- Training Finished ---")
    return history


def evaluate_model(model: torch.nn.Module, device: str, test_loader: torch.utils.data.DataLoader) -> Tuple[List[int], List[int], float]:
    """
    Evaluates the EcoConvNet model and computes inference latency.
    
    Args:
        model: The trained PyTorch model.
        device: 'cuda' or 'cpu'.
        test_loader: DataLoader for the evaluation dataset.
        
    Returns:
        A tuple of (true_labels, predicted_labels, average_latency_ms).
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    total_latency = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            start_time = time.perf_counter()
            output = model(data)
            
            # Synchronize CUDA to accurately measure time if using GPU
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                
            total_latency += (time.perf_counter() - start_time)
            num_samples += data.size(0)
            
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    avg_latency_ms = (total_latency / num_samples) * 1000
    return all_targets, all_preds, avg_latency_ms
