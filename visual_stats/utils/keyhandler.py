from abc import abstractmethod, ABC


class KeyListener(ABC):
    @abstractmethod
    def key_pressed(self, *args, **kwargs):
        pass


class KeyHandler:
    def __init__(self):
        self.listeners = []

    def register(self, listener: KeyListener):
        self.listeners.append(listener)

    def __call__(self, key: str):
        for listener in self.listeners:
            listener.key_pressed(key)
