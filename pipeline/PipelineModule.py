from abc import ABC, abstractmethod


class PipelineModule(ABC):
    def __init__(self):
        self.pipeline = None
        self.name = self.__class__.__name__
        self.outputs = {
            "a": None,
            "b": None,
            "m": None
        }
        print(f"Creating module {self.__class__.__name__}")

    @abstractmethod
    def run(self, a, b, mask, tform, error):
        pass

    def find_root(self):
        # Bubble up the request to the parent pipeline
        if self.pipeline is not None:
            return self.pipeline.find_root()
        else:
            return self

    def get_module(self, name):
        return None

    def get_module_by_type(self, class_name):
        return []