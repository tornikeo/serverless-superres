# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import time
model_inputs = {'image': 'https://i.imgur.com/IdcBNz7.jpg'}
res = requests.post('http://localhost:9000/', json = model_inputs)
print(res.json())

# from main_test_swinir import main
# main()
# from app import init
# init()