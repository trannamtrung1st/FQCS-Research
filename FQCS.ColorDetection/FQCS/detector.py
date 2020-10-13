from .config_model import DetectorConfig

class FQCSDetector:
    def __init__(self, config: DetectorConfig = None):
        self.config = config if config is not None else DetectorConfig()
        
        