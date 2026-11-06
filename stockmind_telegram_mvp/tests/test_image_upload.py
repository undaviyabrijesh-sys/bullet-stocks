import unittest

from stockmind_telegram_mvp.app.image_upload import (
    BoundedBytesIO,
    UploadTooLargeError,
)


class BoundedBytesIOTests(unittest.TestCase):
    def test_accepts_writes_up_to_exact_limit(self):
        destination = BoundedBytesIO(5)
        self.assertEqual(destination.write(b"abc"), 3)
        self.assertEqual(destination.write(b"de"), 2)
        self.assertEqual(destination.getvalue(), b"abcde")

    def test_rejects_overflow_without_partially_writing_chunk(self):
        destination = BoundedBytesIO(5)
        destination.write(b"abc")
        with self.assertRaises(UploadTooLargeError):
            destination.write(b"def")
        self.assertEqual(destination.getvalue(), b"abc")

    def test_rejects_single_chunk_larger_than_limit(self):
        destination = BoundedBytesIO(4)
        with self.assertRaises(UploadTooLargeError):
            destination.write(b"12345")
        self.assertEqual(destination.getvalue(), b"")

    def test_overwrite_within_limit_remains_allowed(self):
        destination = BoundedBytesIO(5)
        destination.write(b"abcde")
        destination.seek(1)
        destination.write(b"Z")
        self.assertEqual(destination.getvalue(), b"aZcde")

    def test_overwrite_that_extends_past_limit_is_rejected(self):
        destination = BoundedBytesIO(5)
        destination.write(b"abcde")
        destination.seek(4)
        with self.assertRaises(UploadTooLargeError):
            destination.write(b"XY")
        self.assertEqual(destination.getvalue(), b"abcde")

    def test_requires_positive_integer_limit(self):
        for invalid in (0, -1, True, 1.5, "5"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    BoundedBytesIO(invalid)


if __name__ == "__main__":
    unittest.main()
