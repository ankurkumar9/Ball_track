from collections import deque

class SimpleTracker:
    def __init__(self, max_history: int = 2000):
        self.last_position = None
        self.history = deque(maxlen=max_history)

    def update(self, detection):
        if detection is not None:
            self.last_position = detection
            self.history.append(detection)
        return self.last_position

    def get_trajectory(self):
        return list(self.history)
