"""Write uploaded video bytes to the OS temp dir; OpenCV needs a path (not project cwd)."""
import os
import tempfile


def write_video_bytes_to_tempfile(content: bytes, suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="isocourt_")
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return path


def safe_unlink(path: str | None) -> None:
    if path and os.path.isfile(path):
        try:
            os.unlink(path)
        except OSError:
            pass
