import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="centered"
)

# Generator Model (same as training script)
class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, hidden_dim=256):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Generator network
        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)
        # Concatenate noise and label embedding
        input_tensor = torch.cat([z, label_embed], dim=1)
        # Generate image
        img = self.model(input_tensor)
        img = img.view(-1, 1, 28, 28)
        return img

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    device = torch.device('cpu')  # Use CPU for web deployment
    generator = Generator()
    
    try:
        # Try to load the saved model
        generator.load_state_dict(torch.load('generator_mnist.pth', map_location=device))
        generator.eval()
        return generator, device
    except FileNotFoundError:
        st.error("Model file 'generator_mnist.pth' not found. Please ensure the model is trained and saved.")
        return None, device

def generate_digit_images(generator, device, digit, num_images=5):
    """Generate multiple images of a specific digit"""
    if generator is None:
        return None
    
    z_dim = 100
    with torch.no_grad():
        z = torch.randn(num_images, z_dim).to(device)
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        generated_images = generator(z, labels)
        
        # Convert to numpy and denormalize
        images = generated_images.cpu().numpy()
        images = (images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        images = np.clip(images, 0, 1)  # Ensure values are in [0, 1]
        
        return images

def create_image_grid(images, digit):
    """Create a grid of generated images"""
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    fig.suptitle(f'Generated images of digit {digit}', fontsize=16, y=1.05)
    
    for i in range(5):
        if i < len(images):
            img = images[i].squeeze()
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'Sample {i+1}', fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# Main app
def main():
    st.title("🔢 Handwritten Digit Image Generator")
    st.markdown("Generate synthetic MNIST-like images using a trained Conditional GAN model.")
    
    # Load model
    generator, device = load_model()
    
    if generator is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        st.info("To use this app, you need to:")
        st.markdown("""
        1. Train the Conditional GAN model using the provided training script
        2. Save the generator as 'generator_mnist.pth' 
        3. Upload the model file to the same directory as this app
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # User interface
    st.markdown("---")
    st.markdown("### Generate Images")
    
    # Digit selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_digit = st.selectbox(
            "Choose a digit to generate (0-9):",
            options=list(range(10)),
            index=2  # Default to digit 2
        )
    
    with col2:
        st.markdown(f"**Selected digit: {selected_digit}**")
    
    # Generate button
    if st.button("🎲 Generate Images", type="primary"):
        with st.spinner("Generating images..."):
            # Generate images
            images = generate_digit_images(generator, device, selected_digit, 5)
            
            if images is not None:
                # Create and display image grid
                image_grid = create_image_grid(images, selected_digit)
                st.image(image_grid, caption=f"Generated images of digit {selected_digit}")
                
                # Display individual images in columns
                st.markdown("### Individual Images")
                cols = st.columns(5)
                for i in range(5):
                    with cols[i]:
                        img = images[i].squeeze()
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax.set_title(f'Sample {i+1}', fontsize=8)
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        plt.close()
                        
                        st.image(Image.open(buf))
            else:
                st.error("Failed to generate images.")
    
    # Model information
    st.markdown("---")
    st.markdown("### Model Information")
    st.info("""
    **Model Architecture:** Conditional Generative Adversarial Network (cGAN)
    
    **Training Data:** MNIST dataset (28x28 grayscale handwritten digits)
    
    **Generator:** Takes random noise + digit label as input, outputs 28x28 images
    
    **Framework:** PyTorch
    """)
    
    # Instructions
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Select a digit** (0-9) from the dropdown menu
        2. **Click "Generate Images"** to create 5 new synthetic images
        3. **View the results** in both grid and individual formats
        
        Each generation uses different random noise, so you'll get unique images every time!
        """)

if __name__ == "__main__":
    main()