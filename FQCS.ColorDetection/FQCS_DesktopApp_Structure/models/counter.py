from app.observer import Subject


class Counter(Subject):
    def __init__(self):
        super().__init__()
        self.count = 0