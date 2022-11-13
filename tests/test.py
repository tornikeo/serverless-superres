# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import time
import utils


while True:
    try:
        model_inputs = {'image_base64': utils.read_b64('./ETH_LR_sm.png')}
        res = requests.post('http://localhost:8000/', json = model_inputs)
    except:
        time.sleep(.5)


print(res.json())

# from main_test_swinir import main
# main()
# from app import init
# init()