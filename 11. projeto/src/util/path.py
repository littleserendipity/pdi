import control.constant as const
import os

def fn_checkpoint():
    return model(const.fn_CHECKPOINT)

def fn_report():
    return model(const.fn_REPORT)

''' General '''

def data(path="", *paths):
    return __general__(os.path.join("..", const.dn_DATA), path, paths)

def out(path="", *paths):
    return __general__(os.path.join("..", const.dn_OUT), path, paths)

def model(path="", *paths):
    return __general__(os.path.join(".", const.dn_MODEL), path, paths)

def __general__(root, path, paths):
    path = os.path.join(root, path)

    for _, x in enumerate(paths):
        path = os.path.join(path, x)

    if not "." in os.path.basename(path):
        os.makedirs(path, exist_ok=True)

    return path