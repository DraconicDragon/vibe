import asyncio
import logging
import re
import sys
import termios
import threading
import tty
from pathlib import Path

import vibe
from vibe.session import InferenceCancelled

MODEL_SOURCE = "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3"
IMAGE_FOLDER = Path("example/images/")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_RED = "\033[91m"
ANSI_MAGENTA = "\033[95m"
ANSI_RESET = "\033[0m"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("vibe").setLevel(logging.INFO)


def _natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _load_images_from_folder(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise RuntimeError(f"Image folder does not exist: {folder}")

    # optional sort here because otherwise we get 1.jpg, 10.jpg, 11.jpg, ... before 2.jpg
    paths = [
        str(path)
        for path in sorted(folder.iterdir(), key=_natural_sort_key)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise RuntimeError(f"No supported images found in: {folder}")
    return paths


def _start_cancel_listener(session: vibe.ModelSession, stop_event: threading.Event) -> threading.Thread | None:
    # Cancel is not async-exclusive: this same cancel API can also be invoked from
    # another thread while a synchronous session.infer(...) call is running.
    # Note: if using infer() (non-async), cancellation still works, but this listener
    # must run in another thread/task while infer() is blocking.
    if not sys.stdin.isatty():
        return None

    def _listen() -> None:
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                ch = sys.stdin.read(1)
                if stop_event.is_set():
                    break
                if ch.lower() == "c":
                    print(f"{ANSI_CYAN}cancelling...{ANSI_RESET}")
                    requested = session.cancel_current_inference()
                    if requested:
                        print(f"{ANSI_YELLOW}cancel requested (waiting for current step to stop){ANSI_RESET}")
                    else:
                        print(f"{ANSI_RED}no active inference to cancel{ANSI_RESET}")
                    break
        except Exception as exc:
            print(f"{ANSI_RED}cancel listener error: {exc}{ANSI_RESET}")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass

    thread = threading.Thread(target=_listen, name="cancel-key-listener", daemon=True)
    thread.start()
    return thread


async def main() -> None:
    _configure_logging()
    image_paths = _load_images_from_folder(IMAGE_FOLDER)

    # Using 'with' is optional but calls session.close() automatically to free resources when done.
    with vibe.load("wd-eva02-large-v3", source=MODEL_SOURCE, auto_download=False, device="cuda") as session:
        # Prepare inputs
        inputs = [(path, path) for path in image_paths]
        stop_cancel_listener = threading.Event()
        listener = _start_cancel_listener(session, stop_cancel_listener)

        if listener is not None:
            print(f"{ANSI_YELLOW}press c to cancel{ANSI_RESET}")

        # Run async inference in batches
        batch_size = min(5, len(inputs))
        try:
            async for chunk in session.infer_async(
                inputs,
                batch_size=batch_size,
                batch_method="auto",
            ):
                for item in chunk:
                    # Result already sorted by score (high to low)
                    score_dict = item.result.as_score_dict()

                    top_3_scores = list(score_dict.items())[:3]

                    # compact single-line output
                    tags_str = ", ".join(f"{tag}:{score:.3f}" for tag, score in top_3_scores)
                    print(f"{item.input_ref} -> {tags_str}")
        except InferenceCancelled:
            print(f"{ANSI_MAGENTA}inference cancelled by user{ANSI_RESET}")
        finally:
            stop_cancel_listener.set()


if __name__ == "__main__":
    asyncio.run(main())
