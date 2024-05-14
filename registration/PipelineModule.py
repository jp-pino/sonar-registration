from abc import ABC, abstractmethod


class PipelineModule(ABC):
    def __init__(self):
        self.pipeline = None
        print(f"Creating module {self.__class__.__name__}")

    @abstractmethod
    def run(self, a, b, mask, tform):
        pass
    
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        return self


