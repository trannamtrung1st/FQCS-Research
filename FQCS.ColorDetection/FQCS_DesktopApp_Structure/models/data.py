from app.observer import Subject


class Data(Subject):
    def __init__(self):
        super().__init__()
        self.x, self.y = 0, 0
