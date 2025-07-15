# ComfyUI-RMBG v2.6.0
#
# This node facilitates background removal using various models, including RMBG-2.0, INSPYRENET, BEN, BEN2, and BIREFNET-HR.
# It utilizes advanced deep learning techniques to process images and generate accurate masks for background removal.
#
# AILab Image and Mask Tools
# This module is specifically designed for ComfyUI-RMBG, enhancing workflows within ComfyUI.
# It offers a collection of utility nodes for efficient handling of images and masks:
#
# 1. Preview Nodes:
#    - Preview: A universal preview tool for both images and masks.
#    - ImagePreview: A specialized preview tool for images.
#    - MaskPreview: A specialized preview tool for masks.
# 
# 2. Image and Mask Processing Nodes:
#    - MaskOverlay: A node for overlaying a mask on an image.
#    - LoadImage: A node for loading images with some Frequently used options.
#    - ImageMaskConvert: Converts between image and mask formats and extracts masks from image channels.
#
# 3. Mask Processing Nodes:
#    - MaskEnhancer: Refines masks through techniques such as blur, smoothing, expansion/contraction, and hole filling.
#    - MaskCombiner: Combines multiple masks using union, intersection, or difference operations.
#
# 4. Image Processing Nodes:
#    - ImageCombiner: Combines foreground and background images with various blending modes and positioning options.
#    - ImageStitch: Stitches multiple images together in various directions.
#    - ImageCrop: Crops an image to a specified size and position.
#    - ICLoRAConcat: Concatenates images with a mask using IC LoRA.
#    - CropObject: Crops an image to the object in the image.
#    - ImageCompare: Compares two images and returns a mask of the differences.
#
# 5. Input Nodes:
#    - ColorInput: A node for inputting colors in various formats.
#
# License: GPL-3.0
# These nodes are crafted to streamline common image and mask operations within ComfyUI workflows.

import os
import random
import folder_paths
import numpy as np
import hashlib
import torch
import cv2
from nodes import MAX_RESOLUTION
from PIL import Image, ImageFilter, ImageOps, ImageSequence, ImageChops, ImageDraw, ImageFont
import torchvision.transforms.functional as T
from comfy.utils import common_upscale
import torch.nn.functional as F
from comfy import model_management
from comfy_extras.nodes_mask import ImageCompositeMasked
from scipy import ndimage

# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2mask(image):
    return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)

def blend_overlay(img_1, img_2):
    arr1 = np.array(img_1).astype(float) / 255.0
    arr2 = np.array(img_2).astype(float) / 255.0
    mask = arr2 < 0.5
    result = np.zeros_like(arr1)
    result[mask] = 2 * arr1[mask] * arr2[mask]
    result[~mask] = 1 - 2 * (1 - arr1[~mask]) * (1 - arr2[~mask])
    return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))

def fill_mask(width, height, mask, box=(0, 0), color=0):
    bg = Image.new("L", (width, height), color)
    bg.paste(mask, box, mask)
    return bg

def empty_image(width, height, batch_size=1):
    return torch.zeros([batch_size, height, width, 3])

def upscale_mask(mask, width, height):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = common_upscale(mask, width, height, 'bicubic', 'disabled')
    mask = mask.squeeze(1)
    return mask

def extract_alpha_mask(image):
    alpha = image[..., 3]
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    if len(alpha.shape) == 4:
        alpha = alpha[:, :, :, 0]
    return alpha.unsqueeze(1) if alpha.ndim == 3 else alpha

def ensure_mask_shape(mask):
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask.squeeze(1)
    return mask

def resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.resize((width, height), Image.Resampling.LANCZOS)

COLOR_PRESETS = {
    "black": "#000000", "white": "#FFFFFF", "red": "#FF0000", "green": "#00FF00", "blue": "#0000FF",
    "yellow": "#FFFF00", "cyan": "#00FFFF", "magenta": "#FF00FF", "gray": "#808080", "silver": "#C0C0C0",
    "maroon": "#800000", "olive": "#808000", "purple": "#800080", "teal": "#008080", "navy": "#000080",
    "orange": "#FFA500", "pink": "#FFC0CB", "brown": "#A52A2A", "violet": "#EE82EE", "indigo": "#4B0082",
    "light_gray": "#D3D3D3", "dark_gray": "#A9A9A9", "light_blue": "#ADD8E6", "dark_blue": "#00008B",
    "light_blue": "#ADD8E6", "dark_blue": "#00008B", "light_green": "#90EE90", "dark_green": "#006400"
}

def fix_color_format(color: str) -> str:
    """Fix color format to valid hex code"""
    if not color:
        return ""
    
    color = color.strip().upper()
    if not color.startswith('#'):
        color = f"#{color}"
        
    color = color[1:]
    if len(color) == 3:
        r, g, b = color[0], color[1], color[2]
        return f"#{r}{r}{g}{g}{b}{b}"
    elif len(color) < 6:
        raise ValueError(f"Invalid color format: {color}")
    elif len(color) > 6:
        color = color[:6]
        
    return f"#{color}"

# Base class for preview
class AILab_PreviewBase:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = ""

    def get_unique_filename(self, filename_prefix):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename_prefix + self.prefix_append
        counter = 1
        while True:
            file = f"{filename}_{counter:04d}.png"
            full_path = os.path.join(self.output_dir, file)
            if not os.path.exists(full_path):
                return full_path, file
            counter += 1

    def save_image(self, image, filename_prefix, prompt=None, extra_pnginfo=None):
        results = []
      
        try:
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 4:  # Batch of images
                    for i in range(image.shape[0]):
                        full_output_path, file = self.get_unique_filename(filename_prefix)
                        img = Image.fromarray(np.clip(image[i].cpu().numpy() * 255, 0, 255).astype(np.uint8))
                        img.save(full_output_path)        
                        results.append({"filename": file, "subfolder": "", "type": self.type})
                else:
                    full_output_path, file = self.get_unique_filename(filename_prefix)
                    img = Image.fromarray(np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8))
                    img.save(full_output_path)
                    results.append({"filename": file, "subfolder": "", "type": self.type})
            else:
                full_output_path, file = self.get_unique_filename(filename_prefix)
                image.save(full_output_path)
                results.append({"filename": file, "subfolder": "", "type": self.type})

            return {
                "ui": {"images": results},
            }
        except Exception as e:
            print(f"Error saving image: {e}")
            return {"ui": {}}

# Preview node
class AILab_Preview(AILab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def preview(self, image=None, mask=None, prompt=None, extra_pnginfo=None):
        results = []
        
        if image is not None:
            image_result = self.save_image(image, "image_preview", prompt, extra_pnginfo)
            if "ui" in image_result and "images" in image_result["ui"]:
                results.extend(image_result["ui"]["images"])
        
        if mask is not None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            mask_result = self.save_image(preview, "mask_preview", prompt, extra_pnginfo)
            if "ui" in mask_result and "images" in mask_result["ui"]:
                results.extend(mask_result["ui"]["images"])
        
        return {
            "ui": {"images": results},
            "result": (image if image is not None else None, mask if mask is not None else None)
        }

# Mask overlay node
class AILab_MaskOverlay(AILab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "mask_opacity": "Control mask opacity (0.0-1.0)",
            "mask_color": "Color for the mask overlay",
            "image": "Input image (RGBA will be converted to RGB)",
            "mask": "Input mask"
        }
        
        return {
            "required": {
                "mask_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["mask_opacity"]}),
                "mask_color": ("COLOR", {"default": "#0000FF", "tooltip": tooltips["mask_color"]}),
             },
            "optional": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),                
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "🧪AILab/🖼️IMAGE"
    OUTPUT_NODE = True

    def hex_to_rgb(self, hex_color):
        """Convert hex color code to RGB values (0-1 range)"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return r, g, b

    def ensure_rgb(self, image):
        """Ensure image is RGB format, convert from RGBA if needed"""
        if image.shape[-1] == 4:
            rgb_image = image[..., :3]
            return rgb_image
        return image

    def execute(self, mask_opacity, mask_color, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
        """Execute image and mask composition"""
        if image is not None:
            image = self.ensure_rgb(image)
        
        preview = None
        
        if mask is not None and image is None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask is None and image is not None:
            preview = image
        elif mask is not None and image is not None:
            mask_adjusted = mask * mask_opacity
            mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3).clone()

            r, g, b = self.hex_to_rgb(mask_color)
            mask_image[:, :, :, 0] = r
            mask_image[:, :, :, 1] = g
            mask_image[:, :, :, 2] = b
            
            preview, = ImageCompositeMasked.composite(self, image, mask_image, 0, 0, True, mask_adjusted)
        
        if preview is None:
            preview = empty_image(64, 64)
            
        if mask is None:
            mask = torch.zeros((1, 64, 64))

        # Save preview for display
        result = self.save_image(preview, filename_prefix, prompt, extra_pnginfo)
        
        # Return both the image and mask for further processing
        return {
            "ui": result["ui"] if "ui" in result else {},
            "result": (preview, mask)
        }

# Mask preview node
class AILab_MaskPreview(AILab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_mask_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",),},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "preview_mask"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def preview_mask(self, mask, prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        result = self.save_image(preview, "mask_preview", prompt, extra_pnginfo)
        return {
            "ui": result["ui"],
            "result": (mask,)
        }

# Image preview node
class AILab_ImagePreview(AILab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_image_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",),},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "preview_image"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def preview_image(self, image, prompt=None, extra_pnginfo=None):
        result = self.save_image(image, "image_preview", prompt, extra_pnginfo)
        return {
            "ui": result["ui"],
            "result": (image,)
        }

# Image mask conversion node
class AILab_ImageMaskConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_channel": (["alpha", "red", "green", "blue"], {"default": "alpha"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "convert"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def convert(self, image=None, mask=None, mask_channel="alpha"):
        # Case 1: No inputs
        if image is None and mask is None:
            empty_image = torch.zeros(1, 3, 64, 64)
            empty_mask = torch.zeros(1, 64, 64)
            return (empty_image, empty_mask)
            
        # Case 2: Only mask input
        if image is None and mask is not None:
            if mask.ndim == 4:
                tensor = mask.permute(0, 2, 3, 1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask)
            elif mask.ndim == 3:
                tensor = mask.unsqueeze(-1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask)
            elif mask.ndim == 2:
                tensor = mask.unsqueeze(0).unsqueeze(-1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask.unsqueeze(0))
            else:
                print(f"Invalid mask shape: {mask.shape}")
                empty_image = torch.zeros(1, 3, 64, 64)
                return (empty_image, mask)
            
        # Case 3: Only image input
        if image is not None and mask is None:
            mask_list = []
            for img in image:
                pil_img = tensor2pil(img)
                pil_img = pil_img.convert("RGBA")
                r, g, b, a = pil_img.split()
                if mask_channel == "red":
                    channel_img = r
                elif mask_channel == "green":
                    channel_img = g
                elif mask_channel == "blue":
                    channel_img = b
                elif mask_channel == "alpha":
                    channel_img = a
                mask = np.array(channel_img.convert("L")).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask)
                mask_list.append(mask_tensor)
            result_mask = torch.stack(mask_list)
            return (image, result_mask)

        if image is not None and mask is not None:
            if mask.ndim == 4:  # [B,C,H,W]
                mask = mask.squeeze(1)  # Convert to [B,H,W]
            return (image, mask)

# Mask enhancer node
class AILab_MaskEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "mask": "Input mask to be processed.",
            "sensitivity": "Adjust the strength of mask detection (higher values result in more aggressive detection).",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "smooth": "Smooth the mask edges (0 for no smoothing, higher values create smoother edges).",
            "fill_holes": "Enable to fill holes in the mask.",
            "invert_output": "Enable to invert the mask output (useful for certain effects)."
        }
        
        return {
            "required": {
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5, "tooltip": tooltips["smooth"]}),
                "fill_holes": ("BOOLEAN", {"default": False, "tooltip": tooltips["fill_holes"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def fill_mask_region(self, mask_pil):
        """Fill holes in the mask"""
        mask_np = np.array(mask_pil)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask_np)
        for contour in contours:
            cv2.drawContours(filled_mask, [contour], 0, 255, -1)  # -1 means fill
        return Image.fromarray(filled_mask)

    def process_mask(self, mask, sensitivity=1.0, mask_blur=0, mask_offset=0, smooth=0.0, 
                    fill_holes=False, invert_output=False):
        processed_masks = []
        
        for mask_item in mask:
            m = mask_item * (1 + (1 - sensitivity))
            m = torch.clamp(m, 0, 1)
            
            if smooth > 0:
                mask_np = m.cpu().numpy()
                binary_mask = (mask_np > 0.5).astype(np.float32)
                blurred_mask = ndimage.gaussian_filter(binary_mask, sigma=smooth)
                final_mask = (blurred_mask > 0.5).astype(np.float32)
                m = torch.from_numpy(final_mask)
            
            if fill_holes:
                mask_pil = tensor2pil(m)
                mask_pil = self.fill_mask_region(mask_pil)
                m = pil2tensor(mask_pil).squeeze(0)
            
            if mask_blur > 0:
                mask_pil = tensor2pil(m)
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                m = pil2tensor(mask_pil).squeeze(0)
            
            if mask_offset != 0:
                mask_pil = tensor2pil(m)
                if mask_offset > 0:
                    for _ in range(mask_offset):
                        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
                else:
                    for _ in range(-mask_offset):
                        mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))
                m = pil2tensor(mask_pil).squeeze(0)
            
            if invert_output:
                m = 1.0 - m
            
            processed_masks.append(m.unsqueeze(0))
        
        return (torch.cat(processed_masks, dim=0),)

# Mask combiner node
class AILab_MaskCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_1": ("MASK",),
                "mode": (["combine", "intersection", "difference"], {"default": "combine"})
            },
            "optional": {
                "mask_2": ("MASK", {"default": None}),
                "mask_3": ("MASK", {"default": None}),
                "mask_4": ("MASK", {"default": None})
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine_masks"

    def combine_masks(self, mask_1, mode="combine", mask_2=None, mask_3=None, mask_4=None):
        try:
            masks = [m for m in [mask_1, mask_2, mask_3, mask_4] if m is not None]
            
            if len(masks) <= 1:
                return (masks[0] if masks else torch.zeros((1, 64, 64), dtype=torch.float32),)
                
            ref_shape = masks[0].shape
            masks = [self._resize_if_needed(m, ref_shape) for m in masks]
            
            if mode == "combine":
                result = torch.maximum(masks[0], masks[1])
                for mask in masks[2:]:
                    result = torch.maximum(result, mask)
            elif mode == "intersection":
                result = torch.minimum(masks[0], masks[1])
            else:
                result = torch.abs(masks[0] - masks[1])
                
            return (torch.clamp(result, 0, 1),)
        except Exception as e:
            print(f"Error in combine_masks: {str(e)}")
            print(f"Mask shapes: {[m.shape for m in masks]}")
            raise e
    
    def _resize_if_needed(self, mask, target_shape):
        try:
            if mask.shape == target_shape:
                return mask
                
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            elif len(mask.shape) == 4:
                mask = mask.squeeze(1)
            
            target_height = target_shape[-2] if len(target_shape) >= 2 else target_shape[0]
            target_width = target_shape[-1] if len(target_shape) >= 2 else target_shape[1]
            
            resized_masks = []
            for i in range(mask.shape[0]):
                mask_np = mask[i].cpu().numpy()
                img = Image.fromarray((mask_np * 255).astype(np.uint8))
                img_resized = img.resize((target_width, target_height), Image.LANCZOS)
                mask_resized = np.array(img_resized).astype(np.float32) / 255.0
                resized_masks.append(torch.from_numpy(mask_resized))
            
            return torch.stack(resized_masks)
            
        except Exception as e:
            print(f"Error in _resize_if_needed: {str(e)}")
            print(f"Input mask shape: {mask.shape}, Target shape: {target_shape}")
            raise e

# Image loader node
class AILab_LoadImage:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'))]
        return {
            "required": {
                "image": (sorted(files) or [""], {"image_upload": True}),
                "mask_channel": (["alpha", "red", "green", "blue"], {"default": "alpha", "tooltip": "Select channel to extract mask from"}),
                "upscale_method": (cls.upscale_methods, {"default": "lanczos", "tooltip": "Method used for resizing the image"}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": "Scale image by this factor (ignored if size > 0)"}),
                "resize_mode": (["longest_side", "shortest_side", "width", "height"], {"default": "longest_side", "tooltip": "Choose how to resize the image"}),
                "size": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Target size for the selected resize mode (0 = keep original size)"}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "WIDTH", "HEIGHT")
    FUNCTION = "load_image"
    OUTPUT_NODE = False

    def load_image(self, image, mask_channel="alpha", upscale_method="lanczos", scale_by=1.0, resize_mode="longest_side", size=0, extra_pnginfo=None):
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            img = Image.open(image_path)
            
            orig_width, orig_height = img.size
            
            resampling_map = {
                "nearest-exact": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "area": Image.BOX,
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS
            }
            resampling = resampling_map.get(upscale_method, Image.LANCZOS)
            
            has_alpha = 'A' in img.getbands()
            if has_alpha and mask_channel == "alpha":
                original_alpha = img.getchannel('A')
            
            img_rgb = img.convert('RGB')
            
            if size > 0:
                if resize_mode == "longest_side":
                    if orig_width >= orig_height:
                        new_width = size
                        new_height = int(orig_height * (size / orig_width))
                    else:
                        new_height = size
                        new_width = int(orig_width * (size / orig_height))
                    img_rgb = img_rgb.resize((new_width, new_height), resampling)
                elif resize_mode == "shortest_side":
                    if orig_width <= orig_height:
                        new_width = size
                        new_height = int(orig_height * (size / orig_width))
                    else:
                        new_height = size
                        new_width = int(orig_width * (size / orig_height))
                    img_rgb = img_rgb.resize((new_width, new_height), resampling)
                elif resize_mode == "width":
                    new_width = size
                    new_height = int(orig_height * (size / orig_width))
                    img_rgb = img_rgb.resize((new_width, new_height), resampling)
                elif resize_mode == "height":
                    new_height = size
                    new_width = int(orig_width * (size / orig_height))
                    img_rgb = img_rgb.resize((new_width, new_height), resampling)
            elif scale_by != 1.0:
                new_width = int(orig_width * scale_by)
                new_height = int(orig_height * scale_by)
                img_rgb = img_rgb.resize((new_width, new_height), resampling)
            
            width, height = img_rgb.size
            
            mask = None
            if mask_channel == "alpha" and has_alpha:
                if (size > 0 or scale_by != 1.0) and 'original_alpha' in locals():
                    mask_img = original_alpha.resize((width, height), resampling)
                    mask = np.array(mask_img).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
            
            output_images = []
            output_masks = []
            
            for i in ImageSequence.Iterator(img_rgb):
                i = ImageOps.exif_transpose(i)
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                
                if i.mode != 'RGB':
                    i = i.convert('RGB')
                
                image = np.array(i).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                
                if mask is not None:
                    output_masks.append(mask.unsqueeze(0))
                elif mask_channel == "red" and 'R' in i.getbands():
                    mask = np.array(i.getchannel('R')).astype(np.float32) / 255.0
                    output_masks.append(torch.from_numpy(mask).unsqueeze(0))
                elif mask_channel == "green" and 'G' in i.getbands():
                    mask = np.array(i.getchannel('G')).astype(np.float32) / 255.0
                    output_masks.append(torch.from_numpy(mask).unsqueeze(0))
                elif mask_channel == "blue" and 'B' in i.getbands():
                    mask = np.array(i.getchannel('B')).astype(np.float32) / 255.0
                    output_masks.append(torch.from_numpy(mask).unsqueeze(0))
                else:
                    output_masks.append(torch.ones((1, height, width), dtype=torch.float32, device="cpu"))
                
                output_images.append(image)
            
            if len(output_images) > 1:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]
            
            mask_image = output_mask.reshape((-1, 1, output_mask.shape[-2], output_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            
            return (output_image, output_mask, mask_image, width, height)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading image: {e}")
            empty_image = torch.zeros(1, 3, 64, 64)
            empty_mask = torch.zeros(1, 64, 64)
            empty_mask_image = empty_mask.reshape((-1, 1, 64, 64)).movedim(1, -1).expand(-1, -1, -1, 3)
            return (empty_image, empty_mask, empty_mask_image, 64, 64)
    
    @classmethod
    def IS_CHANGED(cls, image, mask_channel="alpha", upscale_method="lanczos", scale_by=1.0, resize_mode="longest_side", size=0, extra_pnginfo=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(cls, image, mask_channel="alpha", upscale_method="lanczos", scale_by=1.0, resize_mode="longest_side", size=0, extra_pnginfo=None):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        
        return True

# Image combiner node
class AILab_ImageCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "background": ("IMAGE",),
                "mode": (["normal", "multiply", "screen", "overlay", "add", "subtract"], 
                              {"default": "normal"}),
                "foreground_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "foreground_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "position_x": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "position_y": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "foreground_mask": ("MASK", {"default": None}),
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT")
    FUNCTION = "combine_images"
    
    def combine_images(self, foreground, background, mode="normal", foreground_opacity=1.0, 
                      foreground_scale=1.0, position_x=50, position_y=50, foreground_mask=None):
        if len(foreground.shape) == 3:
            foreground = foreground.unsqueeze(0)
        if len(background.shape) == 3:
            background = background.unsqueeze(0)
        
        batch_size = foreground.shape[0]
        output_images = []
        
        for b in range(batch_size):
            fg_pil = tensor2pil(foreground[b])
            bg_pil = tensor2pil(background[b])
            
            if fg_pil.mode != 'RGBA':
                fg_pil = fg_pil.convert('RGBA')
            
            if foreground_scale != 1.0:
                new_width = int(fg_pil.width * foreground_scale)
                new_height = int(fg_pil.height * foreground_scale)
                fg_pil = fg_pil.resize((new_width, new_height), Image.LANCZOS)
            
            if foreground_mask is not None:
                mask_tensor = foreground_mask[b] if len(foreground_mask.shape) > 2 else foreground_mask
                mask_pil = Image.fromarray(np.uint8(mask_tensor.cpu().numpy() * 255))
                if mask_pil.size != fg_pil.size:
                    mask_pil = mask_pil.resize(fg_pil.size, Image.LANCZOS)
                r, g, b, a = fg_pil.split()
                a = ImageChops.multiply(a, mask_pil)
                fg_pil = Image.merge('RGBA', (r, g, b, a))
            
            fg_w, fg_h = fg_pil.size
            bg_w, bg_h = bg_pil.size
            
            x = int(bg_w * position_x / 100 - fg_w / 2)
            y = int(bg_h * position_y / 100 - fg_h / 2)
            
            new_fg = Image.new('RGBA', (bg_w, bg_h), (0, 0, 0, 0))
            new_fg.paste(fg_pil, (x, y), fg_pil)
            fg_pil = new_fg
            
            if bg_pil.mode != 'RGBA':
                bg_pil = bg_pil.convert('RGBA')
            
            if foreground_opacity < 1.0:
                r, g, b, a = fg_pil.split()
                a = Image.eval(a, lambda x: int(x * foreground_opacity))
                fg_pil = Image.merge('RGBA', (r, g, b, a))
            
            if mode == "normal":
                result = bg_pil.copy()
                result = Image.alpha_composite(result, fg_pil)
            else:
                alpha = fg_pil.split()[3]
                fg_rgb = fg_pil.convert('RGB')
                bg_rgb = bg_pil.convert('RGB')
                
                if mode == "multiply":
                    blended = ImageChops.multiply(fg_rgb, bg_rgb)
                elif mode == "screen":
                    blended = ImageChops.screen(fg_rgb, bg_rgb)
                elif mode == "add":
                    blended = ImageChops.add(fg_rgb, bg_rgb, 1.0)
                elif mode == "subtract":
                    blended = ImageChops.subtract(fg_rgb, bg_rgb, 1.0)
                elif mode == "overlay":
                    blended = blend_overlay(fg_rgb, bg_rgb)
                else:
                    blended = fg_rgb
                
                blended = blended.convert('RGBA')
                r, g, b, _ = blended.split()
                blended = Image.merge('RGBA', (r, g, b, alpha))
                result = bg_pil.copy()
                result = Image.alpha_composite(result, blended)
            
            if result.mode != 'RGB':
                white_bg = Image.new('RGB', result.size, 'white')
                result = Image.alpha_composite(white_bg.convert('RGBA'), result)
                result = result.convert('RGB')
            
            output_images.append(pil2tensor(result))
        
        final_image = torch.cat(output_images, dim=0)
        width = final_image.shape[2]
        height = final_image.shape[1]
        
        return (final_image, width, height)

# Mask extractor node
class AILab_MaskExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["extract_masked_area", "apply_mask", "invert_mask"], {"default": "extract_masked_area"}),
                "background": (["Alpha", "original", "Color"], {"default": "Alpha", "tooltip": "Choose background type"}),
                "background_color": ("COLOR", {"default": "#FFFFFF", "tooltip": "Choose background color (Alpha = transparent)"})
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_masked_area"

    def _prepare_mask(self, mask_np, image_shape):
        try:
            if isinstance(mask_np, torch.Tensor):
                mask_np = mask_np.cpu().numpy()
            mask_np = np.array(mask_np)
            while len(mask_np.shape) > 2 and mask_np.shape[-1] == 1:
                mask_np = mask_np.squeeze(-1)
            while len(mask_np.shape) > 2 and mask_np.shape[0] == 1:
                mask_np = mask_np.squeeze(0)
            if len(mask_np.shape) > 2:
                mask_np = mask_np.squeeze()
            if mask_np.shape != image_shape[:2]:
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            mask_np = mask_np[..., np.newaxis]
            mask_np = np.repeat(mask_np, image_shape[2], axis=2)
            return mask_np
        except Exception as e:
            print(f"Error in _prepare_mask: {str(e)}")
            raise e

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    def extract_masked_area(self, image, mode="extract_masked_area", background="Alpha", background_color="#FFFFFF", mask=None):
        try:
            if mask is None and image.shape[-1] == 4:
                alpha = image[..., 3]
                mask = 1.0 - alpha
                image = image[..., :3]
            elif mask is None:
                mask = torch.ones((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)

            pil_image = tensor2pil(image)
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            mask_np = self._prepare_mask(mask, image_np.shape)
            result_np = np.zeros_like(image_np)
            
            if mode == "extract_masked_area":
                result_np = image_np * mask_np
                if background == "Alpha":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * mask_np
                    result_rgba[:, :, 3] = mask_np[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "original":
                    result_np = image_np * mask_np
                elif background == "Color":
                    r, g, b = self.hex_to_rgb(background_color)
                    result_np = result_np + (1 - mask_np) * np.array([r, g, b])
            
            elif mode == "apply_mask":
                result_np = image_np * mask_np
                if background == "Alpha":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * mask_np
                    result_rgba[:, :, 3] = mask_np[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "original":
                    result_np = image_np * mask_np + image_np * (1 - mask_np)
                elif background == "Color":
                    r, g, b = self.hex_to_rgb(background_color)
                    result_np = result_np + (1 - mask_np) * np.array([r, g, b])
            
            elif mode == "invert_mask":
                result_np = image_np * (1 - mask_np)
                if background == "Alpha":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * (1 - mask_np)
                    result_rgba[:, :, 3] = (1 - mask_np)[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "original":
                    result_np = image_np * (1 - mask_np) + image_np * mask_np
                elif background == "Color":
                    r, g, b = self.hex_to_rgb(background_color)
                    result_np = result_np + mask_np * np.array([r, g, b])
            
            result_pil = Image.fromarray(np.clip(result_np * 255, 0, 255).astype(np.uint8))
            return (pil2tensor(result_pil),)
        except Exception as e:
            print(f"Error in extract_masked_area: {str(e)}")
            raise e

# Image Stitch node
class AILab_ImageStitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "concat_direction": (['right', 'top', 'left', 'bottom'], {"default": 'right'}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch_images"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def stitch_images(self, image1, image2, concat_direction):
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            image1 = image1.repeat(max_batch // image1.shape[0], 1, 1, 1)
            image2 = image2.repeat(max_batch // image2.shape[0], 1, 1, 1)

        if concat_direction in ['right', 'left']:
            # Match heights for horizontal stitching
            h1 = image1.shape[1]
            h2, w2 = image2.shape[1:3]
            aspect = w2 / h2
            
            new_h = h1
            new_w = int(h1 * aspect)
            
            image2 = self._resize(image2, new_w, new_h)
        else:
            # Match widths for vertical stitching
            w1 = image1.shape[2]
            h2, w2 = image2.shape[1:3]
            aspect = h2 / w2
            
            new_w = w1
            new_h = int(w1 * aspect)
            
            image2 = self._resize(image2, new_w, new_h)

        ch1, ch2 = image1.shape[-1], image2.shape[-1]
        if ch1 != ch2:
            if ch1 < ch2:
                image1 = torch.cat((image1, torch.ones((*image1.shape[:-1], ch2-ch1), device=image1.device)), dim=-1)
            else:
                image2 = torch.cat((image2, torch.ones((*image2.shape[:-1], ch1-ch2), device=image2.device)), dim=-1)

        if concat_direction == 'right':
            result = torch.cat((image1, image2), dim=2)
        elif concat_direction == 'bottom':
            result = torch.cat((image1, image2), dim=1)
        elif concat_direction == 'left':
            result = torch.cat((image2, image1), dim=2)
        elif concat_direction == 'top':
            result = torch.cat((image2, image1), dim=1)
            
        return (result,)

    def _resize(self, image, width, height):
        img = image.movedim(-1, 1)
        resized = common_upscale(img, width, height, "lanczos", "disabled")
        return resized.movedim(1, -1)

# Image Crop node
class AILab_ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Width of the crop region in pixels. Will be clamped to image width."}),
                "height": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Height of the crop region in pixels. Will be clamped to image height."}),
                "x_offset": ("INT", {"default": 0, "min": -99999, "step": 1, "tooltip": "Horizontal offset (in pixels) added to the crop position. Positive values move right, negative left."}),
                "y_offset": ("INT", {"default": 0, "min": -99999, "step": 1, "tooltip": "Vertical offset (in pixels) added to the crop position. Positive values move down, negative up."}),
                "split": ("BOOLEAN", {"default": False, "tooltip": "If True, output the cropped region and the rest of the image with the crop area set to zero. If False, the rest is a zero image."}),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"], {"tooltip": "Anchor position for the crop region. Determines where the crop is placed relative to the image."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("CROP", "REST")
    FUNCTION = "execute"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def execute(self, image, width, height, position, x_offset, y_offset, split=False):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)

        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width

        x += x_offset
        y += y_offset

        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        crop = image[:, y:y2, x:x2, :]
        rest = None
        if split:
            top = image[:, 0:y, :, :] if y > 0 else None
            bottom = image[:, y2:oh, :, :] if y2 < oh else None
            left = image[:, y:y2, 0:x, :] if x > 0 else None
            right = image[:, y:y2, x2:ow, :] if x2 < ow else None

            parts = []
            if top is not None:
                parts.append(top)
            if left is not None or right is not None:
                row_parts = []
                if left is not None:
                    row_parts.append(left)
                if right is not None:
                    row_parts.append(right)
                if row_parts:
                    row = torch.cat(row_parts, dim=2)
                    parts.append(row)
            if bottom is not None:
                parts.append(bottom)
            if parts:
                rest = torch.cat(parts, dim=1)
            else:
                rest = torch.zeros_like(image[:, :0, :0, :])
        else:
            rest = image.clone()
            rest[:] = 0
        return (crop, rest)

# ICLoRA Concat node
class AILab_ICLoRAConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "object_image": ("IMAGE",{"tooltip": ("The main image to be used as the foreground (object) in the concatenation.\nIf the image has 4 channels (RGBA), the alpha channel will be automatically extracted and used as the object mask if no mask is provided.")}),
                "layout": (["top-bottom", "left-right"], {"default": "left-right", "tooltip": "The direction in which to concatenate the images: top-bottom or left-right."}),
                "custom_size": ("INT", {"default": 0, "max": MAX_RESOLUTION, "min": 0, "step": 8, "tooltip": "If 0, the output image size is unchanged. Otherwise, sets the base image height (for left-right) or base image width (for top-bottom) in pixels for the concatenation. The object image will be scaled proportionally to match the base image in the concatenation direction."}),
            },
            "optional": {
                "object_mask": ("MASK", {"tooltip": "Mask for the object_image. Defines the region of the object_image to be blended into the base_image."}),
                "base_image": ("IMAGE", {"tooltip": "The background image to be concatenated with the object_image.\nIf the image has 4 channels (RGBA), the alpha channel will be automatically extracted and used as the base mask if no mask is provided."}),
                "base_mask": ("MASK", {"tooltip": "Mask for the base_image. Defines the region of the base_image to be blended with the object_image."}),
            },
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    FUNCTION = "create"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "OBJECT_MASK", "BASE_MASK", "WIDTH", "HEIGHT", "X", "Y")

    def create(self, object_image, layout, custom_size=0, base_image=None, object_mask=None, base_mask=None):
        if object_image.shape[-1] == 4 and object_mask is None:
            object_mask = extract_alpha_mask(object_image)
            object_image = object_image[..., :3]
        if base_image is not None and base_image.shape[-1] == 4 and base_mask is None:
            base_mask = extract_alpha_mask(base_image)
            base_image = base_image[..., :3]

        if base_image is None:
            base_image = empty_image(object_image.shape[2], object_image.shape[1])
            base_mask = torch.full((1, object_image.shape[1], object_image.shape[2]), 1, dtype=torch.float32, device="cpu")
        elif base_image is not None and base_mask is None:
            # raise ValueError("base_mask is required when base_image is provided")
            base_mask = torch.full((1, object_image.shape[1], object_image.shape[2]), 1, dtype=torch.float32, device="cpu")
            
        object_mask = ensure_mask_shape(object_mask)
        base_mask = ensure_mask_shape(base_mask)

        _, base_h, base_w, base_c = base_image.shape
        _, obj_h, obj_w, obj_c = object_image.shape

        if layout == 'left-right':
            if custom_size > 0:
                new_base_h = custom_size
                new_base_w = int(base_w * (custom_size / base_h))
                base_image = base_image.movedim(-1, 1)
                base_image = common_upscale(base_image, new_base_w, new_base_h, 'bicubic', 'disabled')
                base_image = base_image.movedim(1, -1)
                if base_mask is not None:
                    base_mask = upscale_mask(base_mask, new_base_w, new_base_h)
                base_h, base_w = new_base_h, new_base_w

            scale = base_h / obj_h
            new_obj_w = int(obj_w * scale)
            object_image = object_image.movedim(-1, 1)
            object_image = common_upscale(object_image, new_obj_w, base_h, 'bicubic', 'disabled')
            object_image = object_image.movedim(1, -1)
            if object_mask is not None:
                object_mask = upscale_mask(object_mask, new_obj_w, base_h)
            else:
                object_mask = torch.full((1, base_h, new_obj_w), 1, dtype=torch.float32, device="cpu")

            if object_image.shape[-1] != base_image.shape[-1]:
                min_c = min(object_image.shape[-1], base_image.shape[-1])
                object_image = object_image[..., :min_c]
                base_image = base_image[..., :min_c]
            
            image = torch.cat((object_image, base_image), dim=2)
            batch = object_mask.shape[0]
            out_h = base_h
            out_w = new_obj_w + base_w
            object_mask_resized = object_mask
            base_mask_resized = base_mask
            
            if object_mask_resized.shape[-2:] != (base_h, new_obj_w):
                object_mask_resized = upscale_mask(object_mask_resized, new_obj_w, base_h)
            if base_mask_resized.shape[-2:] != (base_h, base_w):
                base_mask_resized = upscale_mask(base_mask_resized, base_w, base_h)
            
            OBJECT_MASK = torch.zeros((batch, out_h, out_w), dtype=object_mask_resized.dtype, device=object_mask_resized.device)
            BASE_MASK = torch.zeros((batch, out_h, out_w), dtype=base_mask_resized.dtype, device=base_mask_resized.device)
            OBJECT_MASK[:, :, :new_obj_w] = object_mask_resized
            BASE_MASK[:, :, new_obj_w:] = base_mask_resized

        elif layout == 'top-bottom':
            if custom_size > 0:
                new_base_w = custom_size
                new_base_h = int(base_h * (custom_size / base_w))
                base_image = base_image.movedim(-1, 1)
                base_image = common_upscale(base_image, new_base_w, new_base_h, 'bicubic', 'disabled')
                base_image = base_image.movedim(1, -1)
                if base_mask is not None:
                    base_mask = upscale_mask(base_mask, new_base_w, new_base_h)
                base_h, base_w = new_base_h, new_base_w

            scale = base_w / obj_w
            new_obj_h = int(obj_h * scale)
            object_image = object_image.movedim(-1, 1)
            object_image = common_upscale(object_image, base_w, new_obj_h, 'bicubic', 'disabled')
            object_image = object_image.movedim(1, -1)
            if object_mask is not None:
                object_mask = upscale_mask(object_mask, base_w, new_obj_h)
            else:
                object_mask = torch.full((1, new_obj_h, base_w), 1, dtype=torch.float32, device="cpu")

            if object_image.shape[-1] != base_image.shape[-1]:
                min_c = min(object_image.shape[-1], base_image.shape[-1])
                object_image = object_image[..., :min_c]
                base_image = base_image[..., :min_c]
            
            image = torch.cat((object_image, base_image), dim=1)
            batch = object_mask.shape[0]
            out_h = new_obj_h + base_h
            out_w = base_w
            object_mask_resized = object_mask
            base_mask_resized = base_mask
            
            if object_mask_resized.shape[-2:] != (new_obj_h, base_w):
                object_mask_resized = upscale_mask(object_mask_resized, base_w, new_obj_h)
            if base_mask_resized.shape[-2:] != (base_h, base_w):
                base_mask_resized = upscale_mask(base_mask_resized, base_w, base_h)
            
            OBJECT_MASK = torch.zeros((batch, out_h, out_w), dtype=object_mask_resized.dtype, device=object_mask_resized.device)
            BASE_MASK = torch.zeros((batch, out_h, out_w), dtype=base_mask_resized.dtype, device=base_mask_resized.device)
            OBJECT_MASK[:, :new_obj_h, :] = object_mask_resized
            BASE_MASK[:, new_obj_h:, :] = base_mask_resized

        x = object_image.shape[2] if layout == 'left-right' else 0
        y = object_image.shape[1] if layout == 'top-bottom' else 0

        return (image, OBJECT_MASK, BASE_MASK, out_w, out_h, x, y)

class AILab_CropObject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "crop_object"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def get_bbox_from_tensor(self, tensor, padding):
        rows = torch.any(tensor > 0, dim=1)
        cols = torch.any(tensor > 0, dim=0)
        if not torch.any(rows) or not torch.any(cols):
            return None
        rmin, rmax = torch.where(rows)[0][[0, -1]]
        cmin, cmax = torch.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - padding)
        rmax = min(tensor.shape[0] - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(tensor.shape[1] - 1, cmax + padding)
        return rmin, rmax, cmin, cmax

    def crop_object(self, image=None, mask=None, padding=0):
        if mask is None and image is None:
            raise ValueError("At least one of image or mask must be provided")
        bbox = None
        if mask is not None:
            mask_tensor = mask.squeeze()
            bbox = self.get_bbox_from_tensor(mask_tensor, padding)
        elif image is not None and image.shape[-1] == 4:
            alpha = image[0, :, :, 3]
            bbox = self.get_bbox_from_tensor(alpha, padding)
        if bbox is None:
            return (image, mask)
        rmin, rmax, cmin, cmax = bbox
        if mask is not None:
            cropped_mask = mask[:, rmin:rmax+1, cmin:cmax+1]
        else:
            if image is not None and image.shape[-1] == 4:
                alpha = image[0, rmin:rmax+1, cmin:cmax+1, 3]
                cropped_mask = alpha.unsqueeze(0)
            else:
                cropped_mask = None
        if image is not None:
            cropped_image = image[:, rmin:rmax+1, cmin:cmax+1, :]
        else:
            cropped_image = None
        return (
            cropped_image if image is not None else image,
            cropped_mask if mask is not None else mask
        )

# Image Compare node
class AILab_ImageCompare:
    def __init__(self):
        self.font_size = 20
        self.padding = 10
        self.bg_color = "white"
        self.font_color = "black"
        self.text_align = "center"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "text1": ("STRING", {"default": "image 1"}),
                "text2": ("STRING", {"default": "image 2"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def get_font(self) -> ImageFont.FreeTypeFont:
        try:
            if os.name == 'nt':
                return ImageFont.truetype("arial.ttf", self.font_size)
            else:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size)
        except:
            base_font = ImageFont.load_default()
            scale_factor = self.font_size / 10
            return ImageFont.TransposedFont(base_font, scale=scale_factor)

    def create_text_panel(self, width: int, text: str) -> Image.Image:
        font = self.get_font()
        
        temp_img = Image.new('RGB', (width, self.font_size * 4), self.bg_color)
        temp_draw = ImageDraw.Draw(temp_img)
        
        text_bbox = temp_draw.textbbox((0, self.font_size), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        final_height = int(text_height * 1.5)
        panel = Image.new('RGB', (width, final_height), self.bg_color)
        draw = ImageDraw.Draw(panel)
        
        x = (width - text_width) // 2
        y = (final_height - text_height) // 2
        
        draw.text((x, y), text, font=font, fill=self.font_color)
        return panel

    def process_image(self, img: Image.Image, target_size: tuple) -> Image.Image:
        target_width, target_height = target_size
        img_width, img_height = img.size
        
        scale_width = target_width / img_width
        scale_height = target_height / img_height
        scale = max(scale_width, scale_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized = resize_image(img, new_width, new_height)
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return resized.crop((left, top, right, bottom))

    def generate(self, image1, image2, text1, text2):
        img1 = tensor2pil(image1)
        img2 = tensor2pil(image2)
        
        if img2.size != img1.size:
            img2 = resize_image(img2, img1.size[0], img1.size[1])
        
        panel1 = None if not text1.strip() else self.create_text_panel(img1.width, text1)
        panel2 = None if not text2.strip() else self.create_text_panel(img2.width, text2)
        
        total_width = img1.width + img2.width + self.padding * 3
        img_height = img1.height
        panel_height = (panel1.height if panel1 else 0) if (panel1 or panel2) else 0
        total_height = img_height + (panel_height + self.padding if panel_height > 0 else 0) + self.padding * 2
        
        result = Image.new('RGB', (total_width, total_height), self.bg_color)

        x1 = self.padding
        x2 = x1 + img1.width + self.padding
        y = self.padding
        
        result.paste(img1, (x1, y))
        result.paste(img2, (x2, y))
        
        if panel1:
            result.paste(panel1, (x1, y + img_height + self.padding))
        if panel2:
            result.paste(panel2, (x2, y + img_height + self.padding))
        
        return (pil2tensor(result),)

# Color Input node
class AILab_ColorInput:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "preset": (list(COLOR_PRESETS.keys()),),
                "color": ("STRING", {"default": "", "placeholder": "Enter color code (e.g. #FF0000 or #F00)"}),
            },
        }

    RETURN_TYPES = ("COLOR",)
    RETURN_NAMES = ("COLOR",)
    FUNCTION = 'get_color'
    CATEGORY = '🧪AILab/🛠️UTIL/🔄IO'

    def get_color(self, preset, color):
        if not color:
            return (COLOR_PRESETS[preset],)
            
        try:
            fixed_color = fix_color_format(color)
            if not all(c in '0123456789ABCDEFabcdef' for c in fixed_color[1:]):
                raise ValueError(f"Invalid hex characters in {color}")
            return (fixed_color,)
        except Exception as e:
            raise RuntimeError(f"Invalid color format: {color}. Please use format like #FF0000 or #F00")

# Image Mask Resize node
class AILab_ImageMaskResize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "image": "Input image to resize",
            "width": "Target width in pixels (0 to keep original width)",
            "height": "Target height in pixels (0 to keep original height)",
            "scale_by": "Scale image by this factor (ignored if width or height > 0)",
            "upscale_method": "Method used for resizing the image",
            "resize_mode": "How to handle aspect ratio: stretch (ignore ratio), resize (maintain ratio by scaling), pad/pad_edge (maintain ratio with padding), crop (maintain ratio by cropping)",
            "pad_color": "Color to use for padding when resize_mode is set to pad",
            "crop_position": "Position to crop from when resize_mode is set to crop",
            "divisible_by": "Make dimensions divisible by this value (useful for some models that require specific dimensions)",
            "mask": "Optional mask to resize along with the image",
            "device": "Device to perform resizing on (CPU or GPU)"
        }
        
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "width": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": tooltips["width"] }),
                "height": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": tooltips["height"] }),
                "scale_by": ("FLOAT", { "default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": tooltips["scale_by"] }),
                "upscale_method": (s.upscale_methods, {"tooltip": tooltips["upscale_method"]}),
                "resize_mode": (["stretch", "resize", "pad", "pad_edge", "crop"], { "default": "stretch", "tooltip": tooltips["resize_mode"] }),
                "pad_color": ("COLOR", { "default": "#FFFFFF", "tooltip": tooltips["pad_color"] }),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center", "tooltip": tooltips["crop_position"] }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, "tooltip": tooltips["divisible_by"] }),
            },
            "optional" : {
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),
                "device": (["cpu", "gpu"], {"default": "cpu", "tooltip": tooltips["device"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT",)
    FUNCTION = "resize"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def resize(self, image, width, height, scale_by, upscale_method, resize_mode, pad_color, crop_position, divisible_by, device="cpu", mask=None):
        B, H, W, C = image.shape

        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        if width == 0 and height == 0:
            if scale_by != 1.0:
                width = int(W * scale_by)
                height = int(H * scale_by)
            else:
                width = W
                height = H
        elif width == 0:
            width = W
        elif height == 0:
            height = H

        new_width = width
        new_height = height
        
        if resize_mode == "resize" or resize_mode.startswith("pad"):
            if width != W or height != H:
                if width == W and height != H:
                    ratio = height / H
                    new_width = round(W * ratio)
                    new_height = height
                elif height == H and width != W:
                    ratio = width / W
                    new_height = round(H * ratio)
                    new_width = width
                else:
                    ratio = min(width / W, height / H)
                    new_width = round(W * ratio)
                    new_height = round(H * ratio)

            if resize_mode.startswith("pad"):
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height

        width = max(1, width)
        height = max(1, height)
        
        if divisible_by > 1:
            width = width - (width % divisible_by) if width >= divisible_by else divisible_by
            height = height - (height % divisible_by) if height >= divisible_by else divisible_by

        out_image = image.clone().to(device)
        if mask is not None:
            out_mask = mask.clone().to(device)
        
        if resize_mode == "crop":
            old_width = W
            old_height = H
            old_aspect = old_width / old_height
            new_aspect = width / height
            
            if old_aspect > new_aspect:
                crop_w = round(old_height * new_aspect)
                crop_h = old_height
            else:
                crop_w = old_width
                crop_h = round(old_width / new_aspect)
            
            if crop_position == "center":
                x = (old_width - crop_w) // 2
                y = (old_height - crop_h) // 2
            elif crop_position == "top":
                x = (old_width - crop_w) // 2
                y = 0
            elif crop_position == "bottom":
                x = (old_width - crop_w) // 2
                y = old_height - crop_h
            elif crop_position == "left":
                x = 0
                y = (old_height - crop_h) // 2
            elif crop_position == "right":
                x = old_width - crop_w
                y = (old_height - crop_h) // 2
            
            out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
            if mask is not None:
                out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)
        
        if (width != W or height != H) or (width != out_image.shape[2] or height != out_image.shape[1]):
            out_image = common_upscale(out_image.movedim(-1,1), width, height, upscale_method, crop="disabled").movedim(1,-1)

            if mask is not None:
                if upscale_method == "lanczos":
                    out_mask = common_upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop="disabled").movedim(1,-1)[:, :, :, 0]
                else:
                    out_mask = common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)
            
        if resize_mode.startswith("pad"):
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height
                
                hex_color = fix_color_format(pad_color)
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
                color = f"{r}, {g}, {b}"
                
                B, H, W, C = out_image.shape
                padded_width = W + pad_left + pad_right
                padded_height = H + pad_top + pad_bottom
                
                bg_color = [int(x.strip())/255.0 for x in color.split(",")]
                if len(bg_color) == 1:
                    bg_color = bg_color * 3
                bg_color = torch.tensor(bg_color, dtype=out_image.dtype, device=out_image.device)
                
                padded_image = torch.zeros((B, padded_height, padded_width, C), dtype=out_image.dtype, device=out_image.device)
                
                for b in range(B):
                    if resize_mode == "pad_edge":
                        top_edge = out_image[b, 0, :, :]
                        bottom_edge = out_image[b, H-1, :, :]
                        left_edge = out_image[b, :, 0, :]
                        right_edge = out_image[b, :, W-1, :]

                        padded_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                        padded_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                        padded_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                        padded_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                    else:
                        padded_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                    
                    padded_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = out_image[b]
                
                if mask is not None:
                    padded_mask = F.pad(
                        out_mask, 
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode='constant',
                        value=0
                    )
                    out_mask = padded_mask
                
                out_image = padded_image

        final_width = out_image.shape[2]
        final_height = out_image.shape[1]
        
        # 创建默认掩码（如果没有提供）
        if mask is None:
            out_mask = torch.zeros((B, final_height, final_width), device=torch.device("cpu"), dtype=torch.float32)
        else:
            out_mask = out_mask.cpu()
            
        return (out_image.cpu(), out_mask, final_width, final_height)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AILab_LoadImage": AILab_LoadImage,
    "AILab_Preview": AILab_Preview,
    "AILab_MaskOverlay": AILab_MaskOverlay,
    "AILab_ImagePreview": AILab_ImagePreview,
    "AILab_MaskPreview": AILab_MaskPreview,
    "AILab_ImageMaskConvert": AILab_ImageMaskConvert,
    "AILab_MaskEnhancer": AILab_MaskEnhancer,
    "AILab_MaskCombiner": AILab_MaskCombiner,
    "AILab_ImageCombiner": AILab_ImageCombiner,
    "AILab_MaskExtractor": AILab_MaskExtractor,
    "AILab_ImageStitch": AILab_ImageStitch,
    "AILab_ImageCrop": AILab_ImageCrop,
    "AILab_ICLoRAConcat": AILab_ICLoRAConcat,
    "AILab_CropObject": AILab_CropObject,
    "AILab_ImageCompare": AILab_ImageCompare,
    "AILab_ColorInput": AILab_ColorInput,
    "AILab_ImageMaskResize": AILab_ImageMaskResize
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_LoadImage": "Load Image (RMBG) 🖼️",
    "AILab_Preview": "Image / Mask Preview (RMBG) 🖼️🎭",
    "AILab_MaskOverlay": "Mask Overlay (RMBG) 🖼️🎭",
    "AILab_ImagePreview": "Image Preview (RMBG) 🖼️",
    "AILab_MaskPreview": "Mask Preview (RMBG) 🎭",
    "AILab_ImageMaskConvert": "Image/Mask Converter (RMBG) 🖼️🎭",
    "AILab_MaskEnhancer": "Mask Enhancer (RMBG) 🎭",
    "AILab_MaskCombiner": "Mask Combiner (RMBG) 🎭",
    "AILab_ImageCombiner": "Image Combiner (RMBG) 🖼️",
    "AILab_MaskExtractor": "Mask Extractor (RMBG) 🎭",
    "AILab_ImageStitch": "Image Stitch (RMBG) 🖼️",
    "AILab_ImageCrop": "Image Crop (RMBG) 🖼️",
    "AILab_ICLoRAConcat": "IC LoRA Concat (RMBG) 🖼️🎭",
    "AILab_CropObject": "Crop To Object (RMBG) 🖼️🎭",
    "AILab_ImageCompare": "Image Compare (RMBG) 🖼️🖼️",
    "AILab_ColorInput": "Color Input (RMBG) 🎨",
    "AILab_ImageMaskResize": "Image Mask Resize (RMBG) 🖼️🎭"
}