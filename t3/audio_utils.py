"""Various audio utilities."""

import sox
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


MODEL = load_silero_vad()


def detect_speech(audio_path: str, threshold: float = 0.8) -> bool:
    """Detect human speech in the audio in `audio_path`, given some confidence `threshold`.

    Args:
        audio_path: Path to the audio file
        threshold: Detection confidence threshold

    Returns:
        Whether a speech was detected or not
    """
    audio = read_audio(audio_path)
    speech_timestamps = get_speech_timestamps(audio, MODEL, threshold=threshold, return_seconds=True)

    return bool(speech_timestamps)


def check_audio_length(audio_path: str, max_length: float = 40) -> bool:
    """Detect if the audio file in `audio_path` is within the desired duration target (<= `max_length`). Too long audio clips would lead into out of memory errors in the S2ST inference.

    Args:
        audio_path: Path to the audio file
        max_length: Maximum permitted length in seconds

    Raises:
        ValueError: When the audio duration could not be determined

    Returns:
        `True` if audio is within the duration limit, `False` if not
    """
    length = sox.file_info.duration(audio_path)

    if length is None:
        raise ValueError(f"'{audio_path}': audio length could not be determined")

    return length <= max_length
