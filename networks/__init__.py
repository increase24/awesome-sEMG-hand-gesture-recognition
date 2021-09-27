#from .XceptionTime import XceptionTime
from .MSCNN import MSCNN

def get_network(name, param):
    model = {'MSCNN':MSCNN}[name]
    return model(param)