import importlib
import control.constant as const
import util.path as path

def train():
    arch = importlib.import_module("%s.%s" % (const.dn_ARCH, const.MODEL))
    arch.model(path.fn_checkpoint())

    print("train")
    
def test():
    print("test")