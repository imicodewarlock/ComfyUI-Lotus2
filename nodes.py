import os
import sys
import torch
import numpy as np
from PIL import Image
import folder_paths

from .pipeline import Lotus2Pipeline
from .infer import load_lora_and_lcm_weights, process_single_image
from diffusers import FlowMatchEulerDiscreteScheduler


class Lotus2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (["depth", "normal"], {"default": "depth"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("LOTUS2_PIPE", "STRING")
    RETURN_NAMES = ("pipe", "task")
    FUNCTION = "load_model"
    CATEGORY = "Lotus2"

    def load_model(self, task, precision):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if precision == "fp16":
            weight_dtype = torch.float16
        elif precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        # 1. Load the core FLUX scheduler and base pipeline
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="scheduler"
        )
        
        pipeline = Lotus2Pipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            scheduler=noise_scheduler,
            torch_dtype=weight_dtype,
        )

        # 2. Inject Lotus-2 Specific LoRA & LCM weights via their built-in loader
        transformer = pipeline.transformer
        transformer, local_continuity_module = load_lora_and_lcm_weights(
            transformer, None, None, None, task
        )
        
        pipeline.transformer = transformer
        pipeline.local_continuity_module = local_continuity_module
        pipeline = pipeline.to(device)

        return (pipeline, task)


class Lotus2Inference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lotus2_pipe": ("LOTUS2_PIPE",),
                "task": ("STRING",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    # Returns: Colorized Visual Image, Raw Array Map (Greyscale Depth or Raw Normals)
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("visual_image", "raw_map")
    FUNCTION = "predict"
    CATEGORY = "Lotus2"

    def predict(self, image, lotus2_pipe, task, seed):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = lotus2_pipe
        
        torch.manual_seed(seed)
        
        visual_outputs = []
        raw_outputs = []
        
        # Process batch images
        for i in range(image.shape[0]):
            # Convert ComfyUI BWHC [0,1] Tensor to standard PIL Image
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Save temporary file since Lotus-2's process_single_image expects an image path
            temp_dir = folder_paths.get_temp_directory()
            temp_path = os.path.join(temp_dir, f"lotus2_input_{i}.png")
            pil_img.save(temp_path)
            
            with torch.inference_mode():
                # process_single_image returns: (output_pred, output_vis, output_sub)
                output_pred, output_vis, _ = process_single_image(
                    temp_path, pipeline, task_name=task, device=device
                )
            
            # Clean up disk
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Process Visual Mapping (Colorized for Depth/Normal)
            vis_np = np.array(output_vis).astype(np.float32) / 255.0
            visual_outputs.append(torch.from_numpy(vis_np))
            
            # Process Raw Numeric Data Mapping
            pred_np = np.array(output_pred).astype(np.float32)
            # Normalize raw data to standard [0, 1] range so ComfyUI nodes can read it natively
            if pred_np.ndim == 2:  # Depth map (H, W) -> expand to (H, W, 1) or (H, W, 3)
                pred_np = np.expand_dims(pred_np, axis=-1)
                pred_np = np.repeat(pred_np, 3, axis=-1) # Create 3-channel grayscale
            
            # Normalize array values safely between 0.0 and 1.0
            p_min, p_max = pred_np.min(), pred_np.max()
            if p_max - p_min > 0:
                pred_np = (pred_np - p_min) / (p_max - p_min)
                
            raw_outputs.append(torch.from_numpy(pred_np))
            
        return (torch.stack(visual_outputs, dim=0), torch.stack(raw_outputs, dim=0))