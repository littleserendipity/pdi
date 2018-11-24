import control.constant as const
import os

''' Dir '''

def dn_aug(sub="", out_dir=False, mkdir=True):
    function = out if out_dir else data
    return function(const.DATASET, const.dn_TRAIN, const.dn_AUGMENTATION, sub, mkdir=mkdir)

def dn_train(sub="", out_dir=False, mkdir=True):
    function = out if out_dir else data
    return function(const.DATASET, const.dn_TRAIN, sub, mkdir=mkdir)

def dn_test(sub="", out_dir=False, mkdir=True):
    function = out if out_dir else data
    return function(const.DATASET, const.dn_TEST, sub, mkdir=mkdir)

''' File '''

def fn_checkpoint():
    return model(const.fn_CHECKPOINT, mkdir=False)

def fn_logger():
    return model(const.fn_LOGGER, mkdir=False)

''' General '''

def join(path, *paths):
    return os.path.join(path, *paths)

def data(path="", *paths, mkdir=True):
    return __general__(os.path.join("..", const.dn_DATA), path, paths, mkdir)

def out(path="", *paths, mkdir=True):
    return __general__(os.path.join("..", const.dn_OUT), path, paths, mkdir)

def model(path="", *paths, mkdir=True):
    return __general__(os.path.join(".", const.dn_MODEL), path, paths, mkdir)

def __general__(root, path, paths, mkdir):
    path = os.path.join(root, path)

    for _, x in enumerate(paths):
        path = os.path.join(path, x)

    if mkdir: os.makedirs(path, exist_ok=True)
    return path