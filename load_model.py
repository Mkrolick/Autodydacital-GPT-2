import torch
from train_gpt2 import GPT


def load_model(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Create a new model instance with the saved config
    model = GPT(checkpoint['config'])
    
    # Load the saved model state
    model.load_state_dict(checkpoint['model'])
    
    # Create optimizer if you plan to continue training
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Move model to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Set to eval mode if doing inference
    model.eval()
    
    return model, optimizer, checkpoint['step']

# Usage example:
model, optimizer, step = load_model('log/ckpt.pt')