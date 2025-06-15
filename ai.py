import argparse
import torch
import random
import os
from diffusers import StableDiffusionXLPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import ImageEnhance

# vocabular enhance
FOOOCUS_VOCAB = [
    "realistic", "depth", "beautiful", "inspiring", "serene", "cinematic", "modern", "natural",
    "dynamic", "stunning", "classic", "coherent", "clean", "dramatic", "contrast", "smooth",
    "professional", "flattering", "atmospheric", "aesthetic", "photo-realistic", "high quality",
    "vibrant", "intricate", "elegant", "soft light", "harmonious", "pleasant", "clear details",
    "dreamy", "thoughtful", "gorgeous", "well-lit", "masterpiece", "detailed", "warm lighting",
    "refined", "glowing", "luminous", "perfect composition", "sharp focus", "award-winning",
    "authentic look", "natural pose", "balanced contrast", "expressive eyes", "real shadows",
    "moody", "iconic", "cozy", "film grain", "portrait lighting", "color graded",
]

def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

def select_contextual_words(prompt, vocab, count=40):
    # aplicare vocabular bazat pe cuvinte
    return [
        word for word in vocab
        if not ("colorful" in word and any(x in prompt for x in ["girl", "woman", "portrait"]))
    ][:count]

def enhance_prompt(original_prompt, vocab, tokenizer, model):
    filtered = select_contextual_words(original_prompt, vocab, count=80)
    selected = random.sample(filtered, 40)
    enhanced = f"{original_prompt}, {', '.join(selected)}"
    
    # maxim 77 cuvinte per prompt
    tokenized = tokenizer(enhanced, truncation=True, max_length=70, return_tensors="pt")
    decoded = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
    
    print(f"Enhanced Prompt (truncated for CLIP):\n{decoded}\n")
    return decoded

def get_available_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename
    filename, extension = os.path.splitext(base_filename)
    counter = 1
    while os.path.exists(f"{filename}_{counter}{extension}"):
        counter += 1
    return f"{filename}_{counter}{extension}"

def calculate_guidance_scale(prompt):
     return round(random.uniform(3.9, 4.7), 1)
    # return 4.0

def apply_sharpness(image, strength=2.1):
    return ImageEnhance.Sharpness(image).enhance(strength)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="realisticStockPhoto_v20.safetensors")
    parser.add_argument("--lora", type=str, default="SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors")
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using {device.type.upper()} backend")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        print(f" Seed set to: {args.seed}")

    # Load model
    print(" Loading model...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.model,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    # Load Lora
    print(f" Loading style LoRA: {args.lora} (strength 0.5)")
    pipe.load_lora_weights(args.lora, weight_name=os.path.basename(args.lora))
    pipe.fuse_lora(lora_scale=0.5)

    # Enhance prompt
    tokenizer, gpt2 = load_gpt2()
    enhanced_prompt = enhance_prompt(args.prompt, FOOOCUS_VOCAB, tokenizer, gpt2)

    # Guidance scale
    guidance = calculate_guidance_scale(args.prompt)
    print(f" Guidance scale: {guidance}")

    # Generare imagine
    print(" Generating image:")
    print(f"[+] {args.prompt}")
    print("[-] ugly, deformed, mutated, blurry, out of frame, bad anatomy, bad hands, extra limbs, poorly drawn face, poorly drawn eyes, bad proportions, low quality, jpeg artifacts, watermark, signature, facial imperfections")

    image = pipe(
        prompt=enhanced_prompt,
        negative_prompt="ugly, deformed, mutated, blurry, out of frame, bad anatomy, bad hands, extra limbs, poorly drawn face, poorly drawn eyes, bad proportions, low quality, jpeg artifacts, watermark, signature, facial imperfections",
        guidance_scale=guidance,
        num_inference_steps=60,
        width=args.width,
        height=args.height
    ).images[0]
    #sharpness pt realism
    image = apply_sharpness(image, 2.1)
    #salvare imagine
    output_path = get_available_filename(args.output)
    image.save(output_path)
    print(f" Saved to {output_path}")

if __name__ == "__main__":
    main()
