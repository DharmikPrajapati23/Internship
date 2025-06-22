import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image

# Load trained generator model (Ensure the model file 'digit_generator.pth' is in the same directory)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

# Load the trained model
generator = Generator()
generator.load_state_dict(torch.load("digit_generator.pth", map_location="cpu"))
generator.eval()

# Streamlit interface
st.title("🖊️ Handwritten Digit Generator")
digit = st.selectbox("Choose a digit to generate (0-9)", list(range(10)))

if st.button("Generate"):
    z = torch.randn(5, 100)  # Random noise for generation
    labels = torch.tensor([digit]*5)  # Same digit label for all 5 images
    
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    
    # Normalize the images to [0, 1] range
    gen_imgs = gen_imgs * 0.5 + 0.5

    # Display the generated images
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(gen_imgs[i][0], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)

# Running the app as a standalone script
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8501))  # Set port for Render
    import streamlit.web.cli as stcli
    import sys
    sys.argv = ["streamlit", "run", "app.py", "--server.port", str(port), "--server.address", "0.0.0.0"]
    sys.exit(stcli.main())
