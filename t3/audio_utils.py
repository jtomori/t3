"""Various audio utilities."""

import sox
import subprocess
import silero_vad


MODEL = silero_vad.load_silero_vad()


def detect_speech(audio_path: str, threshold: float = 0.8) -> bool:
    """Detect human speech in the audio in `audio_path`, given some confidence `threshold`.

    Args:
        audio_path: Path to the audio file
        threshold: Detection confidence threshold

    Returns:
        Whether a speech was detected or not
    """
    audio = silero_vad.read_audio(audio_path)
    speech_timestamps = silero_vad.get_speech_timestamps(audio, MODEL, threshold=threshold, return_seconds=True)

    return bool(speech_timestamps)


def check_audio_length(audio_path: str, max_length: float = 50) -> bool:
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


def convert_mp3_to_ogg(in_path: str, out_path: str, *, quality: int = 0,
                       sampling_rate: int = 22_050, volume: float = 2.3) -> None:
    """TBD"""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-i", in_path, "-c:a", "libvorbis", "-q:a", str(quality), 
           "-ar", str(sampling_rate), "-filter:a", f"volume={volume}", out_path]

    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
