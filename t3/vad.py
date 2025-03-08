"""Voice Activity Detection."""

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


MODEL = load_silero_vad()


def detect_speech(audio_path: str, threshold: float=0.8) -> bool:
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
