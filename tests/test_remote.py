# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import time
import utils
import tempfile
from pathlib import Path

while True:
    try:
        for infile in Path('./tests/data/').glob('*'):
            print('input file', infile)
            model_inputs = {'image_base64': utils.read_b64(infile)}
            res = requests.post('http://localhost:9000/', json = model_inputs)
            ofile = Path('./tests/out/') / infile.name
            utils.write_b64(res.json()['image_base64'], ofile)
            print('output file', ofile)
            input("Continue?")
    except:
        time.sleep(.5)


print(res.json())

# from main_test_swinir import main
# main()
# from app import init
# init()