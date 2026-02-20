import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

# Load data
X_train = torch.FloatTensor(np.load('data/X_train_p.npy'))
y_train = torch.FloatTensor(np.load('data/y_train_p.npy'))
X_test = torch.FloatTensor(np.load('data/X_test_p.npy'))
y_test = torch.FloatTensor(np.load('data/y_test_p.npy'))

# Model definition
class FFNN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=512, output_dim=4):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

model = FFNN()
criterion = nn.MSELoss()
optimizer = optim.RAdam(model.parameters(), lr=1e-3)

# Training configuration
batch_size = 32
train_duration = 120 # seconds
clip_value = 5.0

train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training loop
start_time = time.time()
losses = []
epoch = 0

print(f"Starting training for {train_duration} seconds...")

while (time.time() - start_time) < train_duration:
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        if (time.time() - start_time) >= train_duration:
            break
            
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    epoch += 1
    if epoch % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Elapsed: {elapsed:.2f}s")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test).item()

print(f"Training completed. Final Epochs: {epoch}")
print(f"Test MSE: {test_loss:.6f}")

# Save results
torch.save(model.state_dict(), 'predictive_ffnn.pth')
np.save('predictive_losses.npy', np.array(losses))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Train MSE')
plt.title('FFNN (512 neurons) Training Loss over 120s')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('predictive_error_curve.png')
print("Error curve saved as predictive_error_curve.png")
