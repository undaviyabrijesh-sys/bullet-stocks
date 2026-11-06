"""Bounded in-memory destination for Telegram image downloads."""
from io import BytesIO


class UploadTooLargeError(ValueError):
    """Raised before a download write would exceed the configured byte limit."""


class BoundedBytesIO(BytesIO):
    """A BytesIO that refuses writes which would grow beyond max_bytes."""

    def __init__(self, max_bytes: int):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        super().__init__()
        self.max_bytes = max_bytes

    def write(self, data):
        position = self.tell()
        current_size = self.getbuffer().nbytes
        if max(current_size, position + len(data)) > self.max_bytes:
            raise UploadTooLargeError(
                f"Image download exceeds the {self.max_bytes}-byte limit."
            )
        return super().write(data)
