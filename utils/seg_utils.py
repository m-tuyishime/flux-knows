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
 
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

seg_model = None

def rmbg(img, model_path=None):
    global seg_model
    if not seg_model:
        seg_model = AutoModelForImageSegmentation.from_pretrained(model_path if model_path else "briaai/RMBG-2.0", trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        seg_model.to(f"cuda")
        seg_model.eval()
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_images = transform_image(img).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = seg_model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(img.size)
    white_background = Image.new("RGB", img.size, (255, 255, 255))
    white_background.paste(img, (0, 0), mask)

    return white_background