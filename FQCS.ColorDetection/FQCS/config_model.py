from easydict import EasyDict as edict

class DetectorConfig:
    def __init__(self, 
        name:str = "Detector Configuration",
        detect_range = (0.2,0.8),
        stop_condition = 0,
        min_area = None,
        find_contours_func = "edge"):
        # start
        self.name = name
        self.detect_range = detect_range
        self.stop_condition = 0
        self.min_area = min_area
        self.find_contours_func = find_contours_func

    def get_dict(self):
        data = edict()
        data.name = self.name
        data.detect_range = self.detect_range
        data.stop_condition = self.stop_condition
        data.min_area = self.min_area
        data.find_contours_func = self.find_contours_func
        return data
