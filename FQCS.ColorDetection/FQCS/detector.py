from .config_model import DetectorConfig
import json

class FQCSDetector:
    def __init__(self, config: DetectorConfig = None):
        self.config = config if config is not None else DetectorConfig()
        
    def save_config(self, path):
        with open(path, 'w') as out:
            config_dict = self.config.get_dict()
            json.dump(config_dict, out, indent=2)
            
    def load_config(self, path):
        with open(path) as inp:
            data = json.load(inp)
            self.config = DetectorConfig(**data)
    

