from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# 1) Fetch (with retries + longer timeout) into ./sd2-1-cache
local_dir = snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-1",
    cache_dir="./sd2-1-cache",
    resume_download=True,
    use_auth_token=True
)


# 2) Load your pipeline from that cache
pipe = StableDiffusionPipeline.from_pretrained(
    local_dir,
    torch_dtype=torch.bfloat16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to("cpu")


prompts = [
    "a traider work in degtal assets.",
    "a futuristic buy product using digital currency.",
   
]

results = pipe(
    prompts,
    num_inference_steps=50,
    guidance_scale=3.5,
    height=512,
    width=512
)

images = results.images

# Save or display the images
for i, img in enumerate(images):
    img.save(f"image_{i}.png")  # Save each image