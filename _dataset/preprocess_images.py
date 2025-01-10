import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import shutil
from PIL import Image
from transformers.image_utils import load_image
import sys
sys.path.append('..')
from vision_encoder import ideficsV3
from tqdm import tqdm

class VisionPreprocessor:
    def __init__(self, device=None, param_dtype=torch.float32):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.param_dtype = param_dtype

        # Initialize and freeze the vision encoder
        self.vision_encoder = ideficsV3("HuggingFaceTB/SmolVLM-Instruct").eval().to(self.device)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def load_image(self, image_path):
        """Load an image using PIL without preprocessing."""
        image = load_image(image_path)
        # Convert to tensor without resizing or additional transformations
        inputs = self.vision_encoder.image_processor(images=[image], return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.param_dtype).to(self.device)
        return pixel_values

    def extract_embedding(self, image_tensor):
        """Extract raw vision embedding."""
        with torch.no_grad():
            vision_output = self.vision_encoder(image_tensor)

        vision_output = vision_output.mean(axis=0)

        return vision_output

    def save_embedding(self, vision_output, file_path):
        """Save the vision output to a numpy file."""
        np.save(file_path, vision_output.cpu().numpy())

    def process_directory(self, image_paths, output_dir):
        """Process all images in a directory with a progress bar and save the embeddings."""
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Existing directory cleared: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Adding tqdm for progress bar
        for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
            
            # Load and extract features without preprocessing
            image_tensor = self.load_image(image_path)
            vision_output = self.extract_embedding(image_tensor)

            # Save the output with the same filename but as a .npy
            output_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.npy")
            self.save_embedding(vision_output, output_file_path)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    param_dtype = torch.float32

    # Instantiate the pipeline
    pipeline = VisionPreprocessor(device, param_dtype)

    # Specify input and output directories
    input_directory = "/mnt/nvme/shared_A/datasets/flickr30k/data/flickr30k-images"
    output_directory = "/mnt/nvme/shared_A/datasets/flickr30k/data/vision_embeddings_reduced2"

    image_paths = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Process all images in the input directory
    pipeline.process_directory(image_paths, output_directory)
    print("Processing complete!")
