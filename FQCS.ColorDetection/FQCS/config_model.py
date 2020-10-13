from easydict import EasyDict as edict

class DetectorConfig:
    def __init__(self, 
        name:str = "Detector Configuration"):
        self.name = name

    def get_dict(self):
        data = edict()
        data.name = self.name
        return data
