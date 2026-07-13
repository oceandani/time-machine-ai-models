import os
import argparse
import torch
from PIL import Image
from transformers import pipeline

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline, 
    ControlNetModel,
    EulerDiscreteScheduler
)

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_image(path: str):
    return Image.open(path).convert("RGB")

def resize_keeping_aspect(img: Image.Image, target_w=768, target_h=512):
    w, h = img.size
    target_aspect = target_w / target_h
    img_aspect = w / h
    if abs(img_aspect - target_aspect) < 1e-3:
        new_w, new_h = target_w, target_h
    elif img_aspect > target_aspect:
        new_w = target_w
        new_h = int(round(target_w / img_aspect))
    else:
        new_h = target_h
        new_w = int(round(target_h * img_aspect))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input street photo path")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=334455) # 固定新 Seed
    args = parser.parse_args()

    device = pick_device()
    dtype = torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    print("1. Loading Image & Extracting 3D Depth...")
    init_image = load_image(args.input)
    init_image = resize_keeping_aspect(init_image, target_w=768, target_h=512)

    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
    depth_image = depth_estimator(init_image)["depth"].convert("RGB")

    print("2. Loading High-Resolution AI Models...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "Lykon/dreamshaper-8", 
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if device.type == "mps":
        pipe.enable_attention_slicing()

    lora_filename = "hk_neon_v1.safetensors"
    if os.path.exists(lora_filename):
        print(f"3. Loading Hong Kong LoRA...")
        pipe.load_lora_weights(".", weight_name=lora_filename)
        pipe.fuse_lora(lora_scale=0.4) 

    # 【關鍵改良】加入平坦馬路 (flat and level road) 及 物理描述地標 (curved corner building)
    positive_prompt = (
        "replace the large department store at the right corner with a 1960s Daimaru-style department store facade, "
        "keep the exact storefront placement, exact silhouette and signboard position, "
        "signboard shape and framework similar to the original corner store, "
        "no readable text, no specific letters,"
        "traditional Hong Kong neon/plaque typography style without legible characters"
        "background clutter, random extra objects, extra vehicles, extra buildings, duplicated signs,unrelated text, gibberish characters, floating text"
    )
    
    # 【關鍵改良】強制禁止斜路同幾何扭曲
    negative_prompt = (
        "sloped road, hill, uneven terrain, warped perspective, slanted street, "
        "repetitive architecture, cloned buildings, duplicated patterns, matching facades, "
        "fused people, indistinct faces, blurry cars, melted vehicles, "
        "blurry, soft focus, ghosting, sketch, modern cars, glass skyscrapers"
        "wrong lane direction, wrong road curvature, right turn, change traffic flow"
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print("4. Generating Precision SOGO 1960s Masterpiece...")
    result = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=init_image,               
        control_image=depth_image,      
        strength=0.35,                  # 微調至 0.85：足夠改變年代，但保留足夠原圖馬路平坦的錨點
        num_inference_steps=35,         
        guidance_scale=7.0,
        controlnet_conditioning_scale=0.4, # 降低 Depth 權重至 0.6，防止 AI 過度依賴深度而畫出斜坡
        generator=generator,
    )

    out_path = os.path.join(args.output_dir, "PRECISION_SOGO_1960s.png")
    result.images[0].save(out_path)
    print(f"Success! Precision SOGO intersection saved: {out_path}")

if __name__ == "__main__":
    main()