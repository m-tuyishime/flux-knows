'''
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
 
import yaml
import openai
import base64
import io
import json
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

client = None
gpt_config = {}

def load_gpt_config_from_file(cfg_file="configs/config.yaml"):
    global gpt_config
    with open(cfg_file, 'r') as file:
        config = yaml.safe_load(file)
        gpt_config = config['gpt']


def setup_gpt_config(model_name, provider, base_url=None, api_version=None, ak=None):
    global gpt_config
    global client
    gpt_config["base_url"] = base_url
    gpt_config["api_version"] = api_version
    gpt_config["ak"] = ak
    gpt_config["model_name"] = model_name
    gpt_config["provider"] = provider
    client = None


def init_client():
    global client, gpt_config
    provider = gpt_config.get('provider', 'openai')

    if provider == 'openai':
        base_url = gpt_config['base_url']
        api_version = gpt_config['api_version']
        ak = gpt_config['ak']
        client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            api_version=api_version,
            api_key=ak,
        )
    else:
        model_name = gpt_config['model_name']
        chat_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        client = {
            "processor": chat_processor,
            "model": model,
        }
    return client


@retry(stop=stop_after_attempt(1000), wait=wait_random_exponential(multiplier=1, max=10))
def gpt(image):
    global client, gpt_config
    if not client:
        load_gpt_config_from_file(cfg_file="configs/config.yaml") # TODO
        client = init_client()

    system_prompt = f"""
            When given an image, you make a mosaic image consisting of a 3x3 grid of sub-images showing the exact same subject, describe each subject's appearance in sub-images sequentially from top-left to bottom-right. 
            Limit each description to 50 words. Describe details especially unique appearance like logos, colors, textures, shape, structure and material that can recreate the subject. Refrain from any speculative or guesswork. 

            The output format can refer to the following example but may be changed accordingly to real cases:
            {{
                "row1": {{
                    "image1": "highlights the sneaker's white laces and textured sole, emphasizing its casual style.",
                    "image2": "captures the sneaker's unique color combination and material texture from a slightly angled view.",
                    "image3": "displays a close-up of the sneaker's mint green and lavender panels, focusing on the stitching details."
                }},
                "row2": {{
                    "image1": "presents the sneaker's side, showing the yellow stripe and layered design elements.",
                    "image2": "showcases the sneaker's rounded toe and smooth material finish, highlighting its modern aesthetic.",
                    "image3": "features the sneaker's interior lining and padded collar, emphasizing comfort and design."
                }},
                "row3": {{
                    "image1": "focuses on the sneaker's sole pattern and grip, showcasing its practical features.",
                    "image2": "captures the sneaker's overall shape and color scheme, providing a comprehensive view.",
                    "image3": "features the structure and texture of the subject."
                }},
                "summary": "This set of full-frame photos captures an identical pastel-colored sneaker subject firmly positioned in the real scene, highlighting its unique design, color scheme, and material details from various perspectives (cinematic, epic, 4K, high quality)."
            }}

            The output format can also refer to the following example but may be changed accordingly to real cases:
            {{
                "row1": {{
                    "image1": "captures the overall structure and material of the backpack.",
                    "image2": "captures the backpack's left side, highlighting the gold zipper and smooth fabric texture.",
                    "image3": "displays the right side, emphasizing the Herschel logo patch and badge details."
                }},
                "row2": {{
                    "image1": "focuses on the top section, showcasing the backpack's curved shape and stitching details.",
                    "image2": "presents a close-up of the front pocket, highlighting the zipper and badge designs.",
                    "image3": "shows the backpack's bottom, emphasizing the fabric's color and texture."
                }},
                "row3": {{
                    "image1": "captures the backpack's strap, showing its adjustable buckle and maroon color.",
                    "image2": "highlights the backpack's side pocket, emphasizing the zipper and fabric quality.",
                    "image3": "features the backpack's overall structure, showcasing its sleek design and material."
                }},
                "summary": "This set of full-frame photos captures an identical maroon backpack subject firmly positioned in the real scene, highlighting its unique features, badges, and Herschel logo from various perspectives (cinematic, epic, 4K, high quality)."
            }}

            for content in summary: it should starts with This set of full-frame photos captures an identical xxx subject and include firmly positioned in the real scene.

            Please describe the uploaded image, and return the result in json format.
            Please only return the plain json content without any contextual words, like ```json.
        """
    
    if gpt_config['provider'] == 'openai':
        content = _gpt_openai(image, system_prompt)
    else:
        content = _gpt_huggingface(image, system_prompt)

    # Post processing
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]

    image_info = json.loads(content)
    assert "row1" in image_info and "row2" in image_info and "row3" in image_info and "summary" in image_info, "gpt response format error"
    assert "image1" in image_info["row1"] and "image2" in image_info["row1"] and "image3" in image_info["row1"], "gpt response format error"
    assert "image1" in image_info["row2"] and "image2" in image_info["row2"] and "image3" in image_info["row2"], "gpt response format error"
    assert "image1" in image_info["row3"] and "image2" in image_info["row3"] and "image3" in image_info["row3"], "gpt response format error"
    return image_info

def _gpt_openai(image, system_prompt):
    global client, gpt_config
    model_name = gpt_config['model_name']
    max_tokens = 4096
    temperature = 0.0

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            { "role": "system", "content": system_prompt },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64, {encoded_image}"
                        }
                    }
                ]
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result_json = completion.model_dump_json()
    result_json = json.loads(result_json)
    return result_json["choices"][0]["message"]["content"]

def _gpt_huggingface(image, system_prompt):
    global client
    processor = client["processor"]
    model = client["model"]

    torch.cuda.empty_cache()

    inputs = processor(
        text=system_prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False
        )

        outputs = model.generate(**generation_kwargs)

    return processor.batch_decode(outputs, skip_special_tokens=True)[0]
