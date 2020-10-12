class ColorDetectionConfig:
    def __init__(self):
        self.img_size = (32, 64)
        self.blur_val = 0.05
        self.alpha_r, self.alpha_l = 1, 1
        self.beta_r, self.beta_l = -150, -150
        self.saturation_adj = 2
        self.suppress_thresh = 10
        self.amplify_thresh = (76,31,85)
        self.amplify_rate = 20
        self.max_diff = 0.2

class ErrorDetectionConfig:
    def __init__(self):
