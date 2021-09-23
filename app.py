import os
import gradio as gr
from PIL import Image
import torch

  
os.system('wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth -P experiments/pretrained_models')

def inference(img):
    os.system('mkdir test')
    basewidth = 256
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save("test/1.jpg", "JPEG")
    os.system('python main_test_swinir.py --task real_sr --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq test --scale 4')
    return 'results/swinir_real_sr_x4/1_SwinIR.jpg'
        
title = "Real-ESRGAN"
description = "Gradio demo for Real-ESRGAN. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Please click submit only once"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2107.10833'>Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a> | <a href='https://github.com/xinntao/Real-ESRGAN'>Github Repo</a></p>"
gr.Interface(
    inference, 
    [gr.inputs.Image(type="pil", label="Input")], 
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    enable_queue=True
    ).launch(debug=True)