import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from PIL import Image
import numpy as np

def load_model():
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    # Set config to have only 1 class (binary segmentation for COD10K)
    config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config.num_labels = 1

    # Initialize model with modified config (not from pretrained weights)
    model = SegformerForSemanticSegmentation(config)

    # Load COD10K-trained weights
    checkpoint = torch.load("segformer_cod10k_optimized.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return processor, model

def predict_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # shape: [1, 1, H, W]
    
    # Resize prediction back to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    
    # For binary segmentation (1 class), apply threshold
    predicted = (upsampled_logits.sigmoid() > 0.5).squeeze().byte().cpu().numpy()
    
    # Convert to PIL Image for saving/viewing
    return Image.fromarray(predicted * 255)