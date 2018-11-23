import misc.constant as const
import os

def fn_model_cp():
    return model(const.MODEL_CHECKPOINT)

def fn_model_report():
    return model(const.TXT_REPORT)

def data(path="", *paths):
    return __general__(os.path.join("..", const.F_DATA), path, paths)

def out(path="", *paths):
    return __general__(os.path.join("..", const.F_OUT), path, paths)

def model(path="", *paths):
    return __general__(os.path.join(".", const.F_MODEL), path, paths)

def __general__(root, path, paths):
    path = os.path.join(root, path)

    for _, x in enumerate(paths):
        path = os.path.join(path, x)

    os.makedirs(path, exist_ok=True)
    return path