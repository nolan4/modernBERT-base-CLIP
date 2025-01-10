import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


class ideficsV3(nn.Module):
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct"):
        super().__init__()

        # load smolVLM model from huggingface
        self.image_processor = AutoProcessor.from_pretrained(model_name).image_processor
        smolVLM = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float32)

        # Extract the necessary modules
        self.vision_model = smolVLM.model.vision_model

    def forward(self, pixel_values):

        #################################################################

        # The error ValueError: too many values to unpack (expected 4) occurs because the pixel_values tensor you passed into the model has a shape of [1, 13, 3, 384, 384], while the vision transformer (ViT) expects an input shape of [batch_size, channels, height, width], i.e., a 4D tensor.
        # Your pixel_values tensor is 5D because it contains multiple patches, while the ViT expects a single image or batch of images.
        # You need to flatten the patch dimension (the second dimension, 13) into the batch dimension (1) before passing it to the vision transformer.

        # Flatten the patch dimension into the batch dimension
        batch_size, num_patches, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_patches, channels, height, width)

        #################################################################

        # Run images through the vision transformer
        vision_outputs = self.vision_model(pixel_values)
        x = vision_outputs.last_hidden_state # shape := [batch_size * num_patches, 729, 1152]

        return x
    
if __name__ == "__main__":

    # Instantiate truncated model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    truncated_model = ideficsV3().to(device).eval()
    truncated_model.eval()

    image1 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")
    image2 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")

    inputs1 = truncated_model.image_processor(images=[image1, image2], return_tensors="pt")
    pixel_values = inputs1.pixel_values.to(model_dtype).to(device)

    # Pass pixel_values through your truncated model
    with torch.no_grad():
        outputs = truncated_model(pixel_values)

    print(outputs.shape)  # Should be [batch_size, 2048] given the projection layer output.

