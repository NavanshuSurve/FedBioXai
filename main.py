import torch
import torch_directml
from torch.utils.data import DataLoader
from src.models import FedBioXAI
from src.dataloader import FedBioDataset

# 1. Setup AMD GPU
device = torch_directml.device()
print(f"Running on: {device}")

# 2. Initialize Model
model = FedBioXAI().to(device)

# 3. Load Data for Subject 2 (Initial Test)
dataset = FedBioDataset(subject_id='S2')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Simple Training Loop Snippet
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for batch_idx, batch in enumerate(train_loader):
    # Move only the available signals to GPU
    # This handles your 'permutations' logic automatically
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
    labels = batch['label'].to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

# 5. Save the trained model (for testing on s5)
torch.save(model.state_dict(), 'model_s2.pth')
print("Model saved successfully!")