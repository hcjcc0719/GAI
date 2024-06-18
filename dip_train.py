import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(UNet, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensures output values are in the range [0, 1]
        )
        
        self.skip = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        skip_connection = self.skip(x)
        
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        
        x = self.decoder1(x3)
        x = self.decoder2(x + x2)
        x = self.decoder3(x + x1)
        
        return x + skip_connection

def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def show_image(tensor, title=''):
    tensor = tensor.clamp(0, 1)  # Clamp values to [0, 1] range
    image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the target image
image_path = 'cat.jpg'
image_size = (256, 256) 
target_image = load_image(image_path, image_size).to(device)

# Initialize the DIP model
dip_model = UNet(input_channels=3, output_channels=3).to(device)
optimizer = optim.Adam(dip_model.parameters(), lr=0.001)  # Lower learning rate
criterion = nn.MSELoss()

# Training loop
num_epochs = 3000  # Increase the number of epochs
early_stopping_patience = 300  # Increase patience
best_loss = float('inf')
patience_counter = 0

# Fixed random noise input
input_noise = torch.randn_like(target_image).to(device)

for epoch in range(num_epochs):
    dip_model.train()
    optimizer.zero_grad()
 
    # Forward pass
    output = dip_model(input_noise)
    
    # Compute loss
    loss = criterion(output, target_image)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Early stopping condition
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch}")
        break

    # Print progress
    if epoch % 100 == 0:  # Change to print progress every 100 epochs
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
        show_image(output, title=f'Epoch {epoch}')

# Save the trained model
torch.save(dip_model.state_dict(), 'dip_model.pth')

# Final output image
final_output = output.detach().cpu()
show_image(final_output, title='Final Output')
