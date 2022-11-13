import base64

def read_b64(path) -> str:
    return base64.b64encode(open(path,'rb').read()).decode('utf-8')


def write_b64(b64str, path) -> str:
    open(path,'wb').write(base64.b64decode(b64str.encode('utf-8')))
    return path
