from abc import ABCMeta, abstractmethod


class Base_dataset_image(metaclass=ABCMeta):
    @property
    @abstractmethod
    def cam_id(self):
        pass

    @property
    @abstractmethod
    def img(self):
        pass

    @property
    @abstractmethod
    def image_path(self):
        pass

    @property
    @abstractmethod
    def frame_no_cam(self):
        pass

    @property
    @abstractmethod
    def img_dims(self):
        pass


class Base_dataset:

    def __init__(self):
        pass

    @property
    @abstractmethod
    def __iter__(self):
        pass


    @property
    @abstractmethod
    def __len__(self):
        pass

    @property
    @abstractmethod
    def __next__(self):
        pass