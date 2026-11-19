"""Normalize Python strings to valid Unicode scalar sequences."""


def normalize_unicode_scalars(text: str) -> str:
    """Combine valid surrogate pairs and replace unpaired surrogate code points.

    Some JSON/provider paths can produce explicit UTF-16 surrogate code units
    inside Python strings. Lone surrogates cannot be encoded as UTF-8; replacing
    them with U+FFFD keeps outbound text valid while preserving valid pairs.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = []
    index = 0
    while index < len(text):
        codepoint = ord(text[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(text):
                low = ord(text[index + 1])
                if 0xDC00 <= low <= 0xDFFF:
                    scalar = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
                    normalized.append(chr(scalar))
                    index += 2
                    continue
            normalized.append("\uFFFD")
        elif 0xDC00 <= codepoint <= 0xDFFF:
            normalized.append("\uFFFD")
        else:
            normalized.append(text[index])
        index += 1
    return "".join(normalized)
