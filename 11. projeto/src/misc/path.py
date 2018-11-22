import misc.constant as const
import os

def data(path="", *paths):
    return __general__(os.path.join("..", const.DATA_FOLDER), path, paths)

def out(path="", *paths):
    return __general__(os.path.join("..", const.OUT_FOLDER), path, paths)

def model(path="", *paths):
    return __general__(os.path.join(".", const.MODEL_FOLDER), path, paths)

def fdirModel_CP():
    return model(const.MODEL_CHECKPOINT)

def fdirModel_Report():
    return model(const.TXT_REPORT)

def __general__(root, path, paths):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, path)

    for _, x in enumerate(paths):
        path = os.path.join(path, x)
    return path