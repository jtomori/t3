"""CLI interface, whole pipeline to translate audio in TipToi GME files from German into English."""

import logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

import os
import glob
import shutil
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from . import s2st, audio_utils

log = logging.getLogger("t3")


def main():
    """CLI interface and the whole pipeline."""
    # Arguments parsing
    parser = argparse.ArgumentParser(prog="python -m t3")

    parser.add_argument("input_gme_path", help="Path to the original (to be translated) GME file")
    parser.add_argument("work_dir", help="Path to directory which will contain results and intermediate files")
    parser.add_argument("--skip_translation", action="store_true",
                        help="Skip the translation step (useful for debugging when the translation has happened in previous runs)")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force the inference to happen on CPU, even if a GPU is available (useful for testing on low-end GPUs)")

    args = parser.parse_args()

    # Perform initial checks
    log.info("Performing initial checks")
    checks()

    # Create folder structure
    extracted_dir, translated_dir, final_dir = create_folders(args.work_dir)
    log.info(f"Created folder structure in '{args.work_dir}'")

    # Extract OGG files from GME
    ogg_paths = extract_ogg(args.input_gme_path, extracted_dir)
    log.info(f"Extracted {len(ogg_paths)} files from '{args.input_gme_path}' into '{extracted_dir}'")

    # Split based on audio length (long audio files are songs which don't translate well)
    normal_length, too_long = split_by_length(ogg_paths)
    log.info(f"{len(too_long)} audio files will not be translated because of their duration")

    # Copy too long audio files into the final directory - these will be intact
    [shutil.copy(p, final_dir) for p in too_long]
    log.info(f"Copied too long audios into the '{final_dir}'")

    # Detect speech in the normal length audio files
    speech, sounds = split_by_speech(normal_length)   
    log.info(f"Voice was detected in {len(speech)} files, {len(sounds)} files contain only non-voice sounds")

    # Copy audio files without voice into the final directory - these will be intact
    [shutil.copy(p, final_dir) for p in sounds]
    log.info(f"Copied audios without speech into the '{final_dir}'")

    # Translate normal-length audio files with voice
    if args.skip_translation:
        log.info("Skipping translation (requested with the --skip_translation flag)")
        translated = read_translated_from_disk(speech, translated_dir)
        log.info(f"Loaded {len(translated)} translated audio files from the working directory")

        if not translated:
            raise ValueError("No translated audio files were found (--skip_translation was used)")
    else:
        log.info(f"Translating {len(speech)} audio files")
        translated = s2st.translate_audio_files(speech, translated_dir, force_cpu=args.force_cpu)

    # Convert mp3 into ogg
    convert_to_ogg(translated, final_dir)
    log.info(f"Converted translated mp3 files into ogg files in '{final_dir}'")

    # Build report
    # TODO

    # Insert new audios into the GME
    # TODO


def checks() -> None:
    """Perform initial checks to make sure everything's been correctly set up.

    Raises:
        AssertionError: Indicates a failed check
    """
    assert os.path.exists("SeamlessExpressive"), "The SeamlessExpressive folder was not found in repository's root. Please check README.md for setup instructions."
    assert shutil.which("tttool") is not None, "'tttool' was not found, please check README.md for setup instructions."


def create_folders(work_dir: str) -> None:
    """TBD"""
    extracted_dir = os.path.join(work_dir, "extracted")
    translated_dir = os.path.join(work_dir, "translated")
    final_dir = os.path.join(work_dir, "final")

    for path in [extracted_dir, translated_dir, final_dir]:
        os.makedirs(path, exist_ok=True)

    return extracted_dir, translated_dir, final_dir


def extract_ogg(input_gme_path, extracted_dir) -> list[str]:
    """Extract OGG audio files from the GME file.

    TBD
    """
    cmd = ["tttool", "media", "-d", extracted_dir, input_gme_path]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)

    return glob.glob(os.path.join(extracted_dir, "*.ogg"))


def split_by_length(paths: list[str]) -> tuple[list[str], list[str]]:
    """TBD"""
    ok_length = [audio_utils.check_audio_length(p) for p in paths]
    normal_length = [pair[1] for pair in zip(ok_length, paths) if pair[0]]
    too_long = [pair[1] for pair in zip(ok_length, paths) if not pair[0]]

    return normal_length, too_long


def split_by_speech(paths: list[str]) -> tuple[list[str], list[str]]:
    """TBD"""
    # Use multiple processes to speed this up
    with ProcessPoolExecutor() as e:
        has_voice_it = e.map(audio_utils.detect_speech, paths)

    has_voice = list(has_voice_it)
    speech = [pair[1] for pair in zip(has_voice, paths) if pair[0]]
    sounds = [pair[1] for pair in zip(has_voice, paths) if not pair[0]]

    return speech, sounds


def read_translated_from_disk(paths: list[str], translated_dir: str) -> list[s2st.TranslatedAudio]:
    """TBD"""
    out = []

    for path in paths:
        file_name = os.path.basename(path)
        file_name_base, _ = os.path.splitext(file_name)
        translated_path = os.path.join(translated_dir, f"{file_name_base}.mp3")
        text_path = os.path.join(translated_dir, f"{file_name_base}.txt")

        with open(text_path, encoding="utf-8") as f:
            text = f.read()

        out.append(s2st.TranslatedAudio(translated_path, text))

    return out


def convert_to_ogg(translated: list[s2st.TranslatedAudio], final_dir: str):
    """TBD"""
    with ThreadPoolExecutor() as e:
        for t in translated:
            in_path = t.path

            file_name = os.path.basename(in_path)
            file_name_base, _ = os.path.splitext(file_name)
            out_path = os.path.join(final_dir, f"{file_name_base}.ogg")

            e.submit(audio_utils.convert_mp3_to_ogg, in_path, out_path)


if __name__ == "__main__":
    main()
