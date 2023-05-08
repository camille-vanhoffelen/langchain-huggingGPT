import json


class ConversationHistory:
    def __init__(self):
        self.history = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def __str__(self):
        return json.dumps(self.history)

    def __repr__(self):
        return str(self)
