from pathlib import Path

import zstandard

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_stream(name: str) -> bytes:
    """A raw Beast capture, stored zstd-compressed under tests/fixtures."""
    compressed = (FIXTURES / f"{name}.beast.zst").read_bytes()
    reader = zstandard.ZstdDecompressor().stream_reader(compressed)
    data: bytes = reader.read()
    return data
