import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
from skimage.metrics import peak_signal_noise_ratio

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the UNet architecture for DIP and DDPM
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

# Function to load an image and preprocess
def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# Function to calculate PSNR
def calculate_psnr(image_true, image_generated):
    image_true_np = image_true.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_generated_np = image_generated.squeeze(0).permute(1, 2, 0).cpu().numpy()

    psnr = peak_signal_noise_ratio(image_true_np, image_generated_np)
    return psnr

# Function to display an image
def show_image(tensor, title=''):
    tensor = tensor.clamp(0, 1)  # Clamp values to [0, 1] range
    image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the models
image_size = (256, 256)

# Load the DIP model
dip_model = UNet(input_channels=3, output_channels=3).to(device)
dip_model.load_state_dict(torch.load('dip_model.pth', map_location=device))

# Load the DDPM model
ddpm_model = UNet(input_channels=3, output_channels=3).to(device)
# Adjusted loading for DDPM model
state_dict = torch.load('ddpm_model.pth', map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace('model.', '')] = v
ddpm_model.load_state_dict(new_state_dict)

# Load test image
test_image_path = 'cat.jpg'
test_image = load_image(test_image_path, image_size)

# Evaluate DIP
with torch.no_grad():
    input_noise = torch.randn_like(test_image).to(device)
    dip_output = dip_model(input_noise)
    dip_psnr = calculate_psnr(test_image, dip_output)

# Evaluate DDPM
with torch.no_grad():
    ddpm_output = ddpm_model(test_image)  # Assuming noise level 0 for DDPM
    ddpm_psnr = calculate_psnr(test_image, ddpm_output)

# Evaluate Hybrid Approach (DIP output as DDPM input)
with torch.no_grad():
    dip_output = dip_model(input_noise)
    hybrid_output = ddpm_model(dip_output)  # Assuming noise level 0 for DDPM in hybrid approach
    hybrid_psnr = calculate_psnr(test_image, hybrid_output)

# Measure generation time for DIP
start_time = time.time()
with torch.no_grad():
    input_noise = torch.randn_like(test_image).to(device)
    dip_output = dip_model(input_noise)
end_time = time.time()
dip_generation_time = end_time - start_time

# Measure generation time for DDPM
start_time = time.time()
with torch.no_grad():
    ddpm_output = ddpm_model(test_image)  # Assuming noise level 0 for DDPM
end_time = time.time()
ddpm_generation_time = end_time - start_time

# Measure generation time for Hybrid Approach
start_time = time.time()
with torch.no_grad():
    dip_output = dip_model(input_noise)
    hybrid_output = ddpm_model(dip_output)  # Assuming noise level 0 for DDPM in hybrid approach
end_time = time.time()
hybrid_generation_time = end_time - start_time

# Print results
print("Performance Comparison:")
print("========================")
print(f"DIP: PSNR = {dip_psnr:.2f}, Generation Time = {dip_generation_time:.4f} seconds")
print(f"DDPM: PSNR = {ddpm_psnr:.2f}, Generation Time = {ddpm_generation_time:.4f} seconds")
print(f"Hybrid Approach: PSNR = {hybrid_psnr:.2f}, Generation Time = {hybrid_generation_time:.4f} seconds")

# Show final images
show_image(dip_output, title="DIP Output")
show_image(ddpm_output, title="DDPM Output")
show_image(hybrid_output, title="DDPM Output with DIP Input")
