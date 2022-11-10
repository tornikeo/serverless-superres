# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {'image': 'https://i.imgur.com/dGiiSdo.png'}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())

# from main_test_swinir import main
# main()
# from app import init
# init()