class ConversationMemory:
    def __init__(self):
        self._history = []

    def add_message(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def _count_tokens(self, text: str) -> int:
        return len(text)

    def get_history(self, max_rounds: int = None, max_tokens: int = None) -> list:
        if not self._history:
            return []

        rounds = []
        i = 0
        while i < len(self._history):
            if self._history[i]["role"] == "user":
                round_msgs = [self._history[i]]
                j = i + 1
                while j < len(self._history) and self._history[j]["role"] != "user":
                    round_msgs.append(self._history[j])
                    j += 1
                rounds.append(round_msgs)
                i = j
            else:
                i += 1

        selected_rounds = []
        total_tokens = 0

        for round_msgs in reversed(rounds):
            if max_rounds is not None and len(selected_rounds) >= max_rounds:
                break

            round_tokens = sum(self._count_tokens(msg["content"]) for msg in round_msgs)
            if max_tokens is not None and total_tokens + round_tokens > max_tokens:
                break

            selected_rounds.insert(0, round_msgs)
            total_tokens += round_tokens

        result = []
        for round_msgs in selected_rounds:
            result.extend(round_msgs)

        return result

    def clear(self) -> None:
        self._history.clear()
