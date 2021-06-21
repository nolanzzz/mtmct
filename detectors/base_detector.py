

class Base_detector:
    def __init__(self,cfg):
        self.cfg = cfg
        pass


    def detect(self):
        raise NotImplementedError("Please Implement this method")