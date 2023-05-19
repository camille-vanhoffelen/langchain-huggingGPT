import json


class ConversationHistory:
    """Stores previous user and assistant messages. Used as additional context for task planning."""
    def __init__(self):
        self.history = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def __str__(self):
        return json.dumps(self.history)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.history)

    def __getitem__(self, item):
        return self.history[item]

    def __setitem__(self, key, value):
        self.history[key] = value

    def __delitem__(self, key):
        del self.history[key]

