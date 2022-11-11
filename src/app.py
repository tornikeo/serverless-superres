from transformers import pipeline
import torch
import os
# import gradio as gr
from PIL import Image
from src.main_test_swinir import \
    define_model, get_default_args, get_image_pair, setup
import requests
import tempfile

import numpy as np
import torch
import cv2 
from io import BytesIO
import os.path as osp
import base64


def wget(url: str, path: str) -> str:
    r = requests.get(url, allow_redirects=True)
    print(f'downloading file {url} to {path}')
    open(path, 'wb').write(r.content)
    return path

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global args
    if 'model' not in globals():
        args = get_default_args()
        model = define_model(args)
        model.eval()
        # model = model.half()
        # model = model.half()
        model = model.to('cuda')

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global args
    # Parse out your arguments
    ## Prompt: {"image": "http://something.com/img.jpg"}
    
    with tempfile.TemporaryDirectory() as tmp:
        path = wget(model_inputs['image'], osp.join(tmp, 'image.jpg'))
        folder, save_dir, border, window_size = setup(args)
        _, img_lq, _ = get_image_pair(args, path)
        
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).unsqueeze(0).to('cuda')  # CHW-RGB to NCHW-RGB
        # img_lq = img_lq.half()


        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = model(img_lq)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        cv2.imwrite(osp.join(tmp, 'image_superres.jpg'), output)
        image_base64 = base64.b64encode(open(osp.join(tmp, 'image_superres.jpg'),'rb').read()).decode('utf-8')

    # # Return the results as a dictionary
    return {'image_base64': image_base64}
    # Return the results as a dictionary
# os.system('wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth -P experiments/pretrained_models')

# def inference(img):
#     os.system('mkdir test')
#     basewidth = 256
#     wpercent = (basewidth/float(img.size[0]))
#     hsize = int((float(img.size[1])*float(wpercent)))
#     img = img.resize((basewidth,hsize), Image.ANTIALIAS)
#     img.save("test/1.jpg", "JPEG")
#     os.system('python main_test_swinir.py --task real_sr --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq test --scale 4')
#     return 'results/swinir_real_sr_x4/1_SwinIR.png'
        
# title = "SwinIR"
# description = "Gradio demo for SwinIR. SwinIR achieves state-of-the-art performance on six tasks: image super-resolution (including classical, lightweight and real-world image super-resolution), image denoising (including grayscale and color image denoising) and JPEG compression artifact reduction. See the paper and project page for detailed results below. Here, we provide a demo for real-world image SR.To use it, simply upload your image, or click one of the examples to load them."
# article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2108.10257' target='_blank'>SwinIR: Image Restoration Using Swin Transformer</a> | <a href='https://github.com/JingyunLiang/SwinIR' target='_blank'>Github Repo</a></p>"

# examples=[['ETH_LR.png']]
# gr.Interface(
#     inference, 
#     [gr.inputs.Image(type="pil", label="Input")], 
#     gr.outputs.Image(type="file", label="Output"),
#     title=title,
#     description=description,
#     article=article,
#     enable_queue=True,
#     examples=examples
#     ).launch(debug=True)
