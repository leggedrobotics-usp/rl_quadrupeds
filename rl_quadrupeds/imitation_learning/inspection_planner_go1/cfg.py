class Cfg:
    def __init__(self, **kwargs):
        self._define_defaults()

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown configuration key: {key} in {self.__class__.__name__}")
            setattr(self, key, value)