FROM huggingface/transformers-pytorch-gpu@sha256:6e276dfc6e6cac6a0bb9167e5e7536a975e394b5a7a9907c480affdf3c44640e
RUN apt update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 git wget
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt
WORKDIR /code/app
COPY experiments .
EXPOSE 8000
WORKDIR /code/app
COPY app .
CMD python3 -u app/server.py