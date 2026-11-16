"""Bounded in-memory destination for Telegram image downloads."""
from io import BytesIO


class UploadTooLargeError(ValueError):
    """Raised before a download write would exceed the configured byte limit."""


class BoundedBytesIO(BytesIO):
    """A BytesIO that enforces max_bytes across all size-changing operations."""

    def __init__(self, max_bytes: int):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        super().__init__()
        self.max_bytes = max_bytes

    def write(self, data):
        # len(memoryview) counts elements, not bytes, for formats such as
        # array('I'). BytesIO writes the full buffer byte count, so enforce
        # the limit using nbytes to avoid under-counting multi-byte elements.
        try:
            view = memoryview(data)
        except TypeError:
            # Preserve BytesIO's native TypeError for non-buffer inputs.
            return super().write(data)
        try:
            byte_count = view.nbytes
        finally:
            view.release()

        position = self.tell()
        buffer = self.getbuffer()
        try:
            current_size = buffer.nbytes
        finally:
            buffer.release()
        if max(current_size, position + byte_count) > self.max_bytes:
            raise UploadTooLargeError(
                f"Image download exceeds the {self.max_bytes}-byte limit."
            )
        return super().write(data)

    def writelines(self, lines):
        # BytesIO's inherited C implementation bypasses this subclass's
        # write() override, so route every line through the bounded method.
        for line in lines:
            self.write(line)

