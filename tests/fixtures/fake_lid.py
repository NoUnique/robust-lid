from robust_lid.models import LID


class FakeLID(LID):
    """Test double: returns pre-canned predictions regardless of input."""

    def __init__(self, response: list[tuple[str, float]]) -> None:
        self.response = response
        self.calls: list[str] = []

    def predict(self, text: str) -> list[tuple[str, float]]:
        self.calls.append(text)
        return self.response


class RaisingLID(LID):
    """Test double: raises on predict to exercise error paths."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("boom")

    def predict(self, text: str) -> list[tuple[str, float]]:
        raise self.exc
