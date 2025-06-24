

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToPILImage
from PIL import Image
import io

# --- Streamlit Application Layout (MUST BE FIRST STREAMLIT COMMAND) ---
# Set general page configuration. This must be the very first Streamlit command.
st.set_page_config(
    layout="centered", # or "wide" depending on preference
    page_title="Handwritten Digit Generator",
    initial_sidebar_state="auto"
)

# --- Model Definition (Must be IDENTICAL to the one used during training) ---
# This class must exactly match the Generator class defined in your training script
# (e.g., train_model.py or your Colab notebook) to correctly load the weights.
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        # Embedding layer for the class label
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Main sequential block defining the generator's architecture
        self.main = nn.Sequential(
            # First linear layer to expand the concatenated noise and label embedding
            nn.Linear(noise_dim + num_classes, 128 * (img_size // 4) * (img_size // 4)),
            nn.BatchNorm1d(128 * (img_size // 4) * (img_size // 4)),
            nn.ReLU(True),
            # Reshape the 1D tensor to a 2D feature map for convolutional layers
            nn.Unflatten(1, (128, img_size // 4, img_size // 4)),

            # Transposed Convolutional layers for progressively upsampling the feature maps
            # Each ConvTranspose2d layer increases the spatial dimensions
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Final transposed convolutional layer to reach the target image size (28x28)
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            # Tanh activation scales the output pixel values to the range [-1, 1],
            # matching the normalization applied to MNIST during training.
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise vector with one-hot encoded label (via embedding)
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        img = self.main(gen_input)
        return img

# --- Model Loading Configuration ---
# IMPORTANT: Update this to your new model file name
MODEL_PATH = "digit_generator.pth" 

# These hyperparameters must match those used during training
NOISE_DIM = 100       # Dimension of the latent noise vector
NUM_CLASSES = 10      # Number of classes (digits 0-9)
IMAGE_SIZE = 28       # MNIST image size (28x28)

# --- Model Loading Function ---
# @st.cache_resource decorator caches the loaded model,
# so it's loaded only once across app runs, improving performance.
@st.cache_resource
def load_generator_model(model_path):
    # Set device to CPU for deployment to avoid GPU dependency issues on Render,
    # unless you explicitly configure a GPU instance (which is usually not in free tier).
    device = torch.device("cpu") 
    gen_model = Generator(NOISE_DIM, NUM_CLASSES, IMAGE_SIZE)
    try:
        # Load the state dictionary, mapping to CPU
        gen_model.load_state_dict(torch.load(model_path, map_location=device))
        gen_model.eval() # Set the model to evaluation mode (disables dropout, batchnorm updates)
        return gen_model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. "
                 "Please ensure 'digit_generator.pth' is in the same directory as app.py.")
        return None # Return None if model loading fails

# Load the generator model
generator = load_generator_model(MODEL_PATH)

# --- Streamlit Application Content ---
# Now you can add other Streamlit components below the set_page_config call.
st.title("🔢 Handwritten Digit Generation Web App")
st.markdown("---") # Separator

if generator is None:
    st.warning("Model could not be loaded. Please check the model file path and name.")
else:
    st.write("### Generate Images of Your Desired Digit")

    # User input for digit selection using a Streamlit selectbox
    selected_digit = st.selectbox(
        "**Select the digit (0-9) to generate:**",
        options=list(range(NUM_CLASSES)), # Creates a list [0, 1, ..., 9]
        index=0, # Default selected digit is 0
        help="Choose the digit you want the AI to generate."
    )

    st.markdown("---")

    # Button to trigger image generation
    if st.button("✨ Generate 5 Images"):
        st.write(f"Generating 5 unique images for digit: **{selected_digit}**")
        st.text("Please wait, generating images...")

        generated_images_list = []
        # Disable gradient calculations for inference, saving memory and speeding up.
        with torch.no_grad():
            for i in range(5): # Generate 5 images as required
                # Generate a new random noise vector for each image to ensure diversity
                noise = torch.randn(1, NOISE_DIM, device='cpu') 
                # Create a tensor for the selected digit label
                label = torch.tensor([selected_digit], dtype=torch.long, device='cpu')
                
                # Generate the image using the loaded generator model
                # .cpu() ensures the tensor is on CPU, .squeeze(0) removes the batch dimension
                generated_img_tensor = generator(noise, label).cpu().squeeze(0) 

                # Denormalize the image from [-1, 1] (output of Tanh) to [0, 1] for display purposes
                # This scales pixel values back to a standard image range.
                generated_img_tensor = (generated_img_tensor + 1) / 2 
                
                # Convert the PyTorch tensor image to a PIL Image, which Streamlit can display.
                to_pil = ToPILImage()
                pil_img = to_pil(generated_img_tensor)
                generated_images_list.append(pil_img)

        st.success("Images generated successfully!")
        st.markdown("---")
        st.write(f"### Generated Images for Digit {selected_digit}:")

        # Display images in a grid format similar to the MNIST dataset
        # st.columns creates a row of columns, allowing side-by-side image display.
        cols = st.columns(5) # Create 5 columns for the 5 images
        for i, img in enumerate(generated_images_list):
            with cols[i]: # Place each image in its respective column
                # Use st.image to display the PIL Image
                # CORRECTED: Replaced 'use_column_width=True' with 'use_container_width=True'
                st.image(img, caption=f"Sample {i+1}", use_container_width=True)



# import streamlit as st
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch import nn
# from PIL import Image

# # Load trained generator model (Ensure the model file 'digit_generator.pth' is in the same directory)
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.label_emb = nn.Embedding(10, 10)
#         self.model = nn.Sequential(
#             nn.Linear(110, 128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 784),
#             nn.Tanh()
#         )

#     def forward(self, z, labels):
#         c = self.label_emb(labels)
#         x = torch.cat([z, c], dim=1)
#         img = self.model(x)
#         return img.view(img.size(0), 1, 28, 28)

# # Load the trained model
# generator = Generator()
# generator.load_state_dict(torch.load("digit_generator.pth", map_location="cpu"))
# generator.eval()

# # Streamlit interface
# st.title("🖊️ Handwritten Digit Generator")
# digit = st.selectbox("Choose a digit to generate (0-9)", list(range(10)))

# if st.button("Generate"):
#     z = torch.randn(5, 100)  # Random noise for generation
#     labels = torch.tensor([digit]*5)  # Same digit label for all 5 images
    
#     with torch.no_grad():
#         gen_imgs = generator(z, labels)
    
#     # Normalize the images to [0, 1] range
#     gen_imgs = gen_imgs * 0.5 + 0.5

#     # Display the generated images
#     fig, axs = plt.subplots(1, 5, figsize=(10, 2))
#     for i in range(5):
#         axs[i].imshow(gen_imgs[i][0], cmap='gray')
#         axs[i].axis('off')
#     st.pyplot(fig)

# # Running the app as a standalone script
# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 8501))  # Set port for Render
#     import streamlit.web.cli as stcli
#     import sys
#     sys.argv = ["streamlit", "run", "app.py", "--server.port", str(port), "--server.address", "0.0.0.0"]
#     sys.exit(stcli.main())

