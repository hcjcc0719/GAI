import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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

class DDPM(nn.Module):
    def __init__(self, image_size, in_channels=3, out_channels=3):
        super(DDPM, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = UNet(input_channels=in_channels, output_channels=out_channels)

    def forward(self, x, t):
        return self.model(x)

# Function to create noisy images
def add_noise(img, noise_level):
    noise = torch.randn_like(img) * noise_level
    noisy_img = img + noise
    return noisy_img

# Load the DIP output image as the initial prior\
image_path = 'cat.jpg'
image_size = (256, 256)
initial_prior = load_image(image_path, image_size).to(device)

# Initialize the DDPM model
ddpm_model = DDPM(image_size=image_size, in_channels=3, out_channels=3).to(device)
optimizer = optim.Adam(ddpm_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# DDPM training parameters
num_ddpm_epochs = 1000
diffusion_steps = 1000
noise_levels = torch.linspace(1.0, 0.0, steps=diffusion_steps).to(device)

# Training loop for DDPM
for epoch in range(num_ddpm_epochs):
    ddpm_model.train()
    optimizer.zero_grad()

    noisy_image = add_noise(initial_prior, noise_levels[epoch % diffusion_steps])
    
    # Forward pass
    output = ddpm_model(noisy_image, epoch % diffusion_steps)
    
    # Compute loss
    loss = criterion(output, initial_prior)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Print progress
    if epoch % 100 == 0:
        print(f"DDPM Epoch [{epoch}/{num_ddpm_epochs}], Loss: {loss.item():.4f}")
        show_image(output, title=f'DDPM Epoch {epoch}')

# Save the trained DDPM model
torch.save(ddpm_model.state_dict(), 'ddpm_model.pth')

# Final output image
final_ddpm_output = output.detach().cpu()
show_image(final_ddpm_output, title='Final DDPM Output')
