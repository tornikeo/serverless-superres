FROM huggingface/transformers-pytorch-gpu@sha256:6e276dfc6e6cac6a0bb9167e5e7536a975e394b5a7a9907c480affdf3c44640e
RUN apt update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 git wget
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt
WORKDIR /code/app/experiments/pretrained_models
# COPY experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth .
RUN wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth -P .
RUN wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth -P .
EXPOSE 8000
WORKDIR /code/app/
COPY src src
# CMD python3 -u src/server.py
ENTRYPOINT sanic src.server:server --host 0.0.0.0 --port 8000