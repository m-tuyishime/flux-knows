import os
from PIL import Image
import torch
import glob
import time

from utils.seg_utils import rmbg
from utils.prompt_utils import gpt
from utils.image_utils import resize
from latent_unfold.latent_unfold import LatentUnfoldPipeline
from latent_unfold.register import init_pipeline

config = {
    "base_model":  "black-forest-labs/FLUX.1-dev",
    "seg_model": "briaai/RMBG-2.0"
}

def get_pipeline(aug_att=0.0, grid_shape=(3,3), image_shape=(512,512), cascade=(2,3,4), steps=10):
    global config
    pipe = LatentUnfoldPipeline.from_pretrained( 
        config.get("base_model", "black-forest-labs/FLUX.1-dev"), 
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda", torch.bfloat16)
    pipe = init_pipeline(pipe, aug_att=aug_att, grid_shape=grid_shape, image_shape=image_shape, cascade=cascade, steps=steps) 
    return pipe


if __name__ == "__main__":
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    aug_att = 0.0 # 0.0, 0.02, 0.05
    grid_shape = (3,3) 
    image_shape = (512,512)
    cascade = (2,3) # (2,), (2,3), (2,3,4)
    injection_steps = 14
    pipe = get_pipeline(aug_att=aug_att, grid_shape=grid_shape, image_shape=image_shape, cascade=cascade, steps=injection_steps)

    test_single_view = True
    test_images = []
    use_gpt = False # True is required to set the OpenAI keys in configs/config.yaml
    if test_single_view:
        for p in glob.glob("assets/clock*.jpg"): # single-view
            test_img_rmbg = rmbg(Image.open(p), config.get("seg_model"))
            test_img_rmbg = resize(test_img_rmbg)
            test_images.append(test_img_rmbg)
        user_prompt = "In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall." # prompt comes from OminiControl
        if use_gpt:
            # gpt prompt may introduce random variance
            test_gpt_prompt = gpt(test_images[0])
            short_prompt = f"{test_gpt_prompt['summary']} [IMAGE1] {user_prompt}"
            long_prompt = f"{short_prompt} [IMAGE2]{test_gpt_prompt['row1']['image2']} [IMAGE3]{test_gpt_prompt['row1']['image3']} [IMAGE4]{test_gpt_prompt['row2']['image1']} [IMAGE5]{test_gpt_prompt['row2']['image2']} [IMAGE6]{test_gpt_prompt['row2']['image3']} [IMAGE7]{test_gpt_prompt['row3']['image1']} [IMAGE8]{test_gpt_prompt['row3']['image2']} [IMAGE9]{test_gpt_prompt['row3']['image3']}"
        else:
            long_prompt = f"This set of full-frame photos captures an identical bright yellow alarm clock subject firmly positioned in the real scene, highlighting its retro design, vibrant color, and metal construction from various perspectives (cinematic, epic, 4K, high quality). [IMAGE1]{user_prompt} [IMAGE2]captures the white clock face with black numerals and a large yellow '3' on the right. [IMAGE3]displays the silver metal feet and the glossy yellow finish of the clock's body. [IMAGE4]presents a close-up of the clock's hands, showing the sleek silver design. [IMAGE5]focuses on the clock's side profile, emphasizing the curved yellow frame. [IMAGE6]shows the top view, highlighting the shiny metal handle and bell structure. [IMAGE7]captures the clock's back, revealing the battery compartment and adjustment knobs. [IMAGE8]features the clock's overall shape, showcasing its retro design and vibrant color. [IMAGE9]emphasizes the clock's sturdy metal construction and polished finish."
    else:
        for p in glob.glob("assets/rc_car*.jpg"): # multi-view
            test_img_rmbg = rmbg(Image.open(p), config.get("seg_model"))
            test_img_rmbg = resize(test_img_rmbg)
            test_images.append(test_img_rmbg)
        user_prompt = "A film style shot. On the moon, this item drives across the moon surface. The background is that Earth looms large in the foreground." # prompt comes from OminiControl
        if use_gpt:
            # gpt prompt may introduce random variance
            test_gpt_prompt = gpt(test_images[0])
            short_prompt = f"{test_gpt_prompt['summary']} [IMAGE1] {user_prompt}"
            long_prompt = f"{short_prompt} [IMAGE2]{test_gpt_prompt['row1']['image2']} [IMAGE3]{test_gpt_prompt['row1']['image3']} [IMAGE4]{test_gpt_prompt['row2']['image1']} [IMAGE5]{test_gpt_prompt['row2']['image2']} [IMAGE6]{test_gpt_prompt['row2']['image3']} [IMAGE7]{test_gpt_prompt['row3']['image1']} [IMAGE8]{test_gpt_prompt['row3']['image2']} [IMAGE9]{test_gpt_prompt['row3']['image3']}"
        else:
            long_prompt = f"This set of full-frame photos captures an identical colorful toy car subject firmly positioned in the real scene, highlighting its vibrant design, number '1' badge, and playful features from various perspectives (cinematic, epic, 4K, high quality). [IMAGE1] A film style shot. On the moon, this item drives across the moon surface. The background is that Earth looms large in the foreground. [IMAGE2]captures the toy car's front view, showcasing the number '1' badge and colorful details. [IMAGE3]displays the toy car's side, focusing on the black wheels and number '1' sticker. [IMAGE4]presents the toy car's rear, showing the smooth curves and bright color scheme. [IMAGE5]showcases the toy car's antenna, emphasizing its orange color and blue tip. [IMAGE6]features the toy car's interior, highlighting the blue and red driver figure with a helmet. [IMAGE7]focuses on the toy car's bottom, displaying the green base and textured surface. [IMAGE8]captures the toy car's overall shape and color scheme, providing a comprehensive view. [IMAGE9]features the structure and texture of the subject."
            
    print(long_prompt)

    seed = 1
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    image = pipe(prompt=long_prompt,
                image=test_images, # supports multi-view, single-view
                num_images_per_prompt=1, 
                height=image_shape[0], 
                width=image_shape[1], 
                mask_image=None, 
                strength=1.0, 
                num_inference_steps=28, 
                guidance_scale=7,
                generator=generator,
                mosaic_shape=grid_shape,
                seed=seed).images[0]
    end = time.perf_counter()
    print(f"took {end - start:.6f} seconds")
    image.save(f"{output_dir}/{seed}_{aug_att}.png")