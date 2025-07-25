# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for face parsing using Segformer model.
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from PIL import Image, ImageFilter
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import folder_paths
import shutil
from torchvision import transforms
import requests
import json # Import json module for JSON file validation

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a PyTorch Tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Converts a PyTorch Tensor to a PIL Image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

def image2mask(image: Image.Image) -> torch.Tensor:
    """Converts an image (PIL or Tensor) to a PyTorch mask tensor."""
    if isinstance(image, Image.Image):
        image = pil2tensor(image)
    return image.squeeze()[..., 0]

def mask2image(mask: torch.Tensor) -> Image.Image:
    """Converts a PyTorch mask tensor to a PIL Image."""
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    return tensor2pil(mask)

def RGB2RGBA(image: Image.Image, mask: Union[Image.Image, torch.Tensor]) -> Image.Image:
    """Merges an RGB image with a mask to create an RGBA image."""
    if isinstance(mask, torch.Tensor):
        mask = mask2image(mask)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.merge('RGBA', (*image.convert('RGB').split(), mask.convert('L')))

# Set device to CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Add model folder path to ComfyUI
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

# Centralized management for model download URLs
MODEL_DOWNLOAD_URLS = {
    'config.json': 'https://huggingface.co/1038lab/segformer_face/resolve/main/config.json',
    'model.safetensors': 'https://huggingface.co/1038lab/segformer_face/resolve/main/model.safetensors',
    'preprocessor_config.json': 'https://huggingface.co/1038lab/segformer_face/resolve/main/preprocessor_config.json'
}

class FaceSegment:
    def __init__(self):
        self.processor = None
        self.model = None
        # Model cache directory
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "segformer_face")

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types and parameters for the node."""
        available_classes = [
            "Skin", "Nose", "Eyeglasses", "Left-eye", "Right-eye",
            "Left-eyebrow", "Right-eyebrow", "Left-ear", "Right-ear", "Mouth",
            "Upper-lip", "Lower-lip", "Hair", "Earring", "Neck",
        ]
        tooltips = {
            "process_res": "Processing resolution (higher = more VRAM)",
            "mask_blur": "Blur amount for mask edges",
            "mask_offset": "Expand/Shrink mask boundary",
            "invert_output": "Invert both image and mask output",
            "background": "Choose background type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Choose background color (ignored in Alpha mode)"
        }
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                **{cls_name: ("BOOLEAN", {"default": False})
                   for cls_name in available_classes},
                "process_res": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLOR", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment_face"
    CATEGORY = "🧪AILab/🧽RMBG"

    def check_model_cache(self):
        """Checks if model files exist and are valid in the cache."""
        if not os.path.exists(self.cache_dir):
            return False, "Model directory not found."

        required_files = list(MODEL_DOWNLOAD_URLS.keys())

        missing_or_corrupted_files = []
        for f in required_files:
            file_path = os.path.join(self.cache_dir, f)
            if not os.path.exists(file_path):
                missing_or_corrupted_files.append(f)
                continue

            # For JSON files, try to load to validate integrity
            if f.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_json:
                        json.load(f_json)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[FaceSegment] Warning: File {f} exists but is corrupted ({e}), will attempt to re-download.")
                    missing_or_corrupted_files.append(f)
            else:
                # For safetensors files, simply check file size
                if os.path.getsize(file_path) < 1024: # Assume valid file is at least 1KB
                    print(f"[FaceSegment] Warning: File {f} exists but is too small, possibly corrupted, will attempt to re-download.")
                    missing_or_corrupted_files.append(f)

        if missing_or_corrupted_files:
            return False, f"Missing or corrupted model files: {', '.join(missing_or_corrupted_files)}"
        return True, "Model cache verified."

    def clear_model(self):
        """Clears the loaded model to free up VRAM."""
        if self.model is not None:
            print("[FaceSegment] Clearing model...")
            self.model.cpu()
            del self.model
            self.model = None
            self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[FaceSegment] Model cleared.")


    def download_model_files(self):
        """Downloads model files from predefined URLs."""
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"[FaceSegment] Downloading face parsing model files to: {self.cache_dir}...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        for filename, url in MODEL_DOWNLOAD_URLS.items():
            target_path = os.path.join(self.cache_dir, filename)

            # Check if file exists and is valid, delete if invalid
            if os.path.exists(target_path):
                if filename.endswith('.json'):
                    try:
                        with open(target_path, 'r', encoding='utf-8') as f_json:
                            json.load(f_json)
                        print(f"[FaceSegment] File {filename} already exists and is valid, skipping download.")
                        continue
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        print(f"[FaceSegment] File {filename} exists but is corrupted, attempting re-download.")
                        os.remove(target_path)
                else: # For non-JSON files, simple size check
                    if os.path.getsize(target_path) > 1024: # Assume valid file is at least 1KB
                        print(f"[FaceSegment] File {filename} already exists, skipping download.")
                        continue
                    else:
                        print(f"[FaceSegment] File {filename} exists but is too small, possibly corrupted, attempting re-download.")
                        os.remove(target_path)

            print(f"[FaceSegment] Downloading {filename} from {url}...")
            try:
                with requests.get(url, stream=True, headers=headers, timeout=30) as r:
                    r.raise_for_status() # Check if the HTTP request was successful

                    if filename.endswith('.json'):
                        content = r.text
                        try:
                            json.loads(content) # Validate if the downloaded content is valid JSON
                        except json.JSONDecodeError as e:
                            raise RuntimeError(f"Downloaded content for {filename} is not valid JSON: {e}")
                        with open(target_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    else:
                        with open(target_path, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)

                if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                    print(f"[FaceSegment] File {filename} downloaded successfully.")
                else:
                    raise RuntimeError(f"File {filename} was not downloaded or is empty.")

            except requests.exceptions.RequestException as e:
                print(f"[FaceSegment] Network error or download timeout: {url} - {e}")
                if os.path.exists(target_path):
                    try: os.remove(target_path) # Clean up potentially incomplete files
                    except: pass
                return False, f"Failed to download {filename}, please check your network connection: {str(e)}"
            except Exception as e:
                print(f"[FaceSegment] An error occurred while downloading {filename}: {e}")
                if os.path.exists(target_path):
                    try: os.remove(target_path) # Clean up potentially incomplete files
                    except: pass
                return False, f"Failed to download {filename}: {str(e)}"
        return True, "All model files downloaded successfully."

    @torch.no_grad() # Disable gradient calculation to save memory
    def segment_face(self, images, process_res=512, mask_blur=0, mask_offset=0, background="Alpha", background_color="#222222", invert_output=False, **class_selections):
        """Performs semantic segmentation of faces."""
        try:
            # Check and download model if needed
            cache_status, message = self.check_model_cache()
            if not cache_status:
                print(f"[FaceSegment] Cache check result: {message}")
                download_status, download_message = self.download_model_files()
                if not download_status:
                    raise RuntimeError(download_message)

            # Load model if not already loaded
            if self.processor is None or self.model is None:
                print(f"[FaceSegment] Loading Segformer model from {self.cache_dir}...")
                self.processor = SegformerImageProcessor.from_pretrained(self.cache_dir)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.cache_dir)
                self.model.eval() # Set to evaluation mode
                for param in self.model.parameters():
                    param.requires_grad = False # Freeze model parameters
                self.model.to(device) # Move model to the specified device
                print("[FaceSegment] Model loaded.")

            # Class mapping for face segmentation
            class_map = {
                "Background": 0, "Skin": 1, "Nose": 2, "Eyeglasses": 3,
                "Left-eye": 4, "Right-eye": 5, "Left-eyebrow": 6, "Right-eyebrow": 7,
                "Left-ear": 8, "Right-ear": 9, "Mouth": 10, "Upper-lip": 11,
                "Lower-lip": 12, "Hair": 13, "Hat": 14, "Earring": 15,
                "Necklace": 16, "Neck": 17, "Clothing": 18
            }

            # Get user-selected classes
            selected_classes = [name for name, selected in class_selections.items() if selected]
            if not selected_classes:
                # If no specific classes are selected, default to common face parts
                print("[FaceSegment] No specific face regions selected. Defaulting to: Skin, Nose, Left-eye, Right-eye, Mouth.")
                selected_classes = ["Skin", "Nose", "Left-eye", "Right-eye", "Mouth"]

            # Validate selected classes
            invalid_classes = [cls for cls in selected_classes if cls not in class_map]
            if invalid_classes:
                raise ValueError(f"Invalid class selections: {', '.join(invalid_classes)}. Valid classes are: {', '.join(class_map.keys())}")

            # Image preprocessing pipeline
            transform_image = transforms.Compose([
                transforms.Resize((process_res, process_res)),
                transforms.ToTensor(),
            ])

            batch_tensor = [] # To store processed images
            batch_masks = []  # To store generated masks

            for image in images:
                orig_image = tensor2pil(image)
                w, h = orig_image.size # Original image dimensions

                input_tensor = transform_image(orig_image)

                if input_tensor.shape[0] == 4: # If RGBA image, take only RGB channels
                    input_tensor = input_tensor[:3]

                # Normalize the input tensor
                input_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_tensor)

                input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

                # Perform model inference
                outputs = self.model(input_tensor)
                logits = outputs.logits.cpu() # Move logits back to CPU
                # Upsample logits to original image dimensions
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=(h, w), # Target size is original image height and width
                    mode="bilinear",
                    align_corners=False,
                )
                pred_seg = upsampled_logits.argmax(dim=1)[0] # Get the predicted segmentation map

                # Combine masks for selected classes
                combined_mask = None
                for class_name in selected_classes:
                    mask = (pred_seg == class_map[class_name]).float() # Create binary mask based on class ID
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = torch.clamp(combined_mask + mask, 0, 1) # Combine masks and clamp to 0-1 range

                # Convert mask to PIL Image for further processing
                mask_image = Image.fromarray((combined_mask.numpy() * 255).astype(np.uint8))

                # Apply mask blur
                if mask_blur > 0:
                    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))

                # Apply mask offset
                if mask_offset != 0:
                    if mask_offset > 0:
                        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=mask_offset * 2 + 1))
                    else:
                        mask_image = mask_image.filter(ImageFilter.MinFilter(size=-mask_offset * 2 + 1))

                # Invert output mask
                if invert_output:
                    mask_image = Image.fromarray(255 - np.array(mask_image))

                # Handle background color or transparency
                if background == "Alpha":
                    rgba_image = RGB2RGBA(orig_image, mask_image)
                    result_image = pil2tensor(rgba_image)
                else: # Color mode
                    def hex_to_rgba(hex_color):
                        hex_color = hex_color.lstrip('#')
                        if len(hex_color) == 6:
                            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                            a = 255 # Default to opaque
                        elif len(hex_color) == 8:
                            r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
                        else:
                            raise ValueError("Invalid color format, please use #RRGGBB or #RRGGBBAA.")
                        return (r, g, b, a)
                    rgba_image = RGB2RGBA(orig_image, mask_image)
                    rgba = hex_to_rgba(background_color)
                    bg_image = Image.new('RGBA', orig_image.size, rgba)
                    # Composite the foreground image (processed image) with the background
                    composite_image = Image.alpha_composite(bg_image, rgba_image)
                    result_image = pil2tensor(composite_image.convert('RGB')) # Convert to RGB for output

                batch_tensor.append(result_image)
                batch_masks.append(pil2tensor(mask_image))

            # Create mask images for visualization
            mask_images_for_output = []
            for mask_tensor in batch_masks:
                # Expand single-channel mask to 3-channel RGB image for ComfyUI visualization
                mask_image_rgb = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                mask_images_for_output.append(mask_image_rgb)

            # Stack all image and mask tensors into batches
            final_images_output = torch.cat(batch_tensor, dim=0)
            final_masks_output = torch.cat(batch_masks, dim=0)
            final_mask_image_output = torch.cat(mask_images_for_output, dim=0)

            return (final_images_output, final_masks_output, final_mask_image_output)

        except Exception as e:
            # Catch any processing errors, clear the model, and re-raise the exception
            print(f"[FaceSegment] Error: Face parsing processing failed: {str(e)}")
            self.clear_model() # Clear model to avoid VRAM leaks
            raise RuntimeError(f"Face parsing processing failed: {str(e)}")
        finally:
            # Clear the model after processing (if the model is not in training mode)
            # Note: If you want the model to stay loaded in memory for faster consecutive processing,
            # you can remove this finally block. However, for VRAM-sensitive systems, clearing
            # after each process can be beneficial.
            if self.model is not None and not self.model.training:
                self.clear_model()

# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "FaceSegment": FaceSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSegment": "Face Segment (RMBG)"
}
