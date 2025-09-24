import torch
import os


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, is_best=False):
    """
    Save a complete training checkpoint

    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model.get_config() if hasattr(model, 'get_config') else {}
    }

    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, path)

    # Optionally save best model separately
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, scheduler=None, path='checkpoint.pth', device='cuda'):
    """
    Load a complete training checkpoint

    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        scheduler: Learning rate scheduler (optional)
        path: Path to checkpoint file
        device: Device to load on ('cuda' or 'cpu')

    Returns:
        epoch: The epoch number
        loss: The loss value when saved
    """
    checkpoint = torch.load(path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Set model to training mode
    model.train()

    return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))