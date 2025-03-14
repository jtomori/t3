"""CLI interface, whole pipeline to translate audio in TipToi GME files from German into English."""

import logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

import os
import csv
import sox
import glob
import shutil
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from . import s2st, audio_utils

log = logging.getLogger("t3")


def main() -> None:
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
    checks(args)

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
    [shutil.copy(p, final_dir) for p in too_long]  # pylint: disable=expression-not-assigned
    log.info(f"Copied too long audios into the '{final_dir}'")

    # Detect speech in the normal length audio files
    speech, sounds = split_by_speech(normal_length)
    log.info(f"Voice was detected in {len(speech)} files, {len(sounds)} files contain only non-voice sounds")

    # Copy audio files without voice into the final directory - these will be intact
    [shutil.copy(p, final_dir) for p in sounds]  # pylint: disable=expression-not-assigned
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
    report_path = csv_report(ogg_paths, too_long, speech, translated, args.work_dir)
    log.info(f"Saved a CSV report into '{report_path}'")

    # Create filelist
    filelist_path = prepare_filelist(extracted_dir, final_dir)
    log.info(f"Created filelist '{filelist_path}'")

    # Insert new audios into the GME
    new_gme_path = create_gme(args.work_dir, filelist_path, args.input_gme_path)
    log.info(f"Created translated GME '{new_gme_path}'")
    log.info("Done")


def checks(args: argparse.Namespace) -> None:
    """Perform initial checks to make sure everything's been correctly set up.

    Args:
        args: Parsed CLI arguments

    Raises:
        AssertionError: Indicates a failed check
    """
    assert os.path.exists("SeamlessExpressive"), "The SeamlessExpressive folder was not found in repository's root. Please check README.md for setup instructions."
    assert os.path.exists("libtiptoi"), "./libtiptoi was not detected, please compile it. Check README.md for setup instructions."

    # libtiptoi breaks with spaces in paths
    if " " in args.work_dir:
        log.warning(f"Space was detected in the '{args.work_dir}' argument, it's being replaced with underscore")
        args.work_dir = args.work_dir.replace(" ", "_")


def create_folders(work_dir: str) -> tuple[str, str, str]:
    """Create directory structure which will hold intermediate files.

    Args:
        work_dir: Location of the working directory which will contain all intermediate, resulting files

    Returns:
        Tuple with 3 paths: directory for extracted OGG files, translated MP3 files and the folder for final files to be assembled back into a GME
    """
    extracted_dir = os.path.join(work_dir, "extracted")
    translated_dir = os.path.join(work_dir, "translated")
    final_dir = os.path.join(work_dir, "final")

    for path in [extracted_dir, translated_dir, final_dir]:
        os.makedirs(path, exist_ok=True)

    return extracted_dir, translated_dir, final_dir


def extract_ogg(input_gme_path: str, extracted_dir: str) -> list[str]:
    """Extract OGG audio files into `extracted_dir` from the GME file `input_gme_path`.

    Note that libtiptoi does not set error codes properly, so this function might silently fail in some cases.

    Args:
        input_gme_path: Path to the input GME file
        extracted_dir: Path where OGG files will be extracted into

    Returns:
        List of paths of the extracted OGG files
    """
    cmd = ["./libtiptoi", "x", f"{extracted_dir}/", input_gme_path]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        # libtiptoi returns 1 upon success
        if exc.returncode == 1:
            pass

    return glob.glob(os.path.join(extracted_dir, "*.ogg"))


def split_by_length(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split audio files in `paths` based on their length into two lists which are returned.

    This is useful to filter out long files - songs (which don't translate well), or file which would
    cause out of memory errors during the translation.

    Args:
        paths: Input paths

    Returns:
        `normal_length, too_long` - a tuple of two lists of paths, `too_long` containing paths to files which were above the duration limit
    """
    ok_length = [audio_utils.check_audio_length(p) for p in paths]
    normal_length = [pair[1] for pair in zip(ok_length, paths) if pair[0]]
    too_long = [pair[1] for pair in zip(ok_length, paths) if not pair[0]]

    return normal_length, too_long


def split_by_speech(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split audio files in `paths` based on the voice activity detection (VAD) into two lists which are returned.

    This is useful for isolating audio files with (can be translated) and without (shouldn't be translated) human speech.
    The VAD is not perfect, so some files might be miscategorised.

    Args:
        paths: Input paths

    Returns:
        `speech, sounds` - a tuple of two lists of paths, `speech` containing paths to files with detected speech
    """
    # Use multiple processes to speed this up
    with ProcessPoolExecutor() as e:
        has_voice_it = e.map(audio_utils.detect_speech, paths)

    has_voice = list(has_voice_it)
    speech = [pair[1] for pair in zip(has_voice, paths) if pair[0]]
    sounds = [pair[1] for pair in zip(has_voice, paths) if not pair[0]]

    return speech, sounds


def read_translated_from_disk(paths: list[str], translated_dir: str) -> list[s2st.TranslatedAudio]:
    """Read already translated audio files from the disk in `translated_dir` (produced by a previous run).

    Args:
        paths: List of extracted OGG files which are expected to have been translated (this list is expected to contain only files with speech)
        translated_dir: Path to the directory containing translated files

    Returns:
        List of audio path & transcribed text pairs, matching `s2st.translate_audio_files`'s return signature
    """
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


def convert_to_ogg(translated: list[s2st.TranslatedAudio], final_dir: str) -> None:
    """Convert translated mp3 files in `translated` into TipToi-compatible OGG files, in `final_dir`.

    Args:
        translated: Paths to files to be converted into OGG
        final_dir: Destination directory path
    """
    with ThreadPoolExecutor() as e:
        for t in translated:
            in_path = t.path

            file_name = os.path.basename(in_path)
            file_name_base, _ = os.path.splitext(file_name)
            out_path = os.path.join(final_dir, f"{file_name_base}.ogg")

            e.submit(audio_utils.convert_mp3_to_ogg, in_path, out_path)


def csv_report(ogg_paths: list[str], too_long: list[str], speech: list[str], translated: list[s2st.TranslatedAudio], work_dir: str) -> str:  # pylint: disable=too-many-locals
    """Create a CSV report in `work_dir` with info about each audio file's duration, category and transcript.

    Args:
        ogg_paths: List of paths of all extracted OGG files
        too_long: List of paths of files which have been classified as too long
        speech: List of paths of files which have been classified to contain voice
        translated: List of paths, text pairs of translated files
        work_dir: Path of the working directory, where the CSV report will be saved into

    Returns:
        File path of the CSV report
    """
    # Create rows
    rows = []

    for ogg in sorted(ogg_paths):
        filename = os.path.basename(ogg)
        name, _ = os.path.splitext(filename)

        duration = sox.file_info.duration(ogg)

        if ogg in too_long:
            category = "Too long"
        elif ogg in speech:
            category = "Speech"
        else:
            category = "Sound"

        text = ""
        for tr in translated:
            if name in tr.path:
                text = tr.text
                break

        rows.append((name, duration, category, text))

    # Save into CSV
    csv_path = os.path.join(work_dir, "report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        csv_writer = csv.writer(f, delimiter=",")

        csv_writer.writerow(["OGG file", "Duration", "Category", "Transcript"])  # Header
        csv_writer.writerows(rows)  # Data

    return csv_path


def prepare_filelist(extracted_dir: str, final_dir: str) -> str:
    """Create a filelist needed for replacing audio files in the input GME file.

    The file list is based on the one created during the OGG extraction step (`extract_ogg()`).

    Args:
        extracted_dir: Location of extracted OGG files and filelist
        final_dir: Location of the translated audio files and where the new filelist will be saved into

    Returns:
        Path to the new filelist
    """
    path_src = os.path.join(extracted_dir, "filelist.txt")
    path_dst = os.path.join(final_dir, "filelist.txt")

    # Read original list
    with open(path_src, encoding="utf-8") as f:
        filelist_src = f.readlines()

    # Update paths
    filelist_dst = [line.replace(extracted_dir, final_dir) for line in filelist_src]

    # Save the updated list to the new path
    with open(path_dst, "w", encoding="utf-8") as f:
        f.writelines(filelist_dst)

    return path_dst


def create_gme(workdir_path: str, filelist_path: str, input_gme_path: str) -> str:
    """Create a new TipToi GME file with translated audio files specified in `filelist_path`, the GME file is derived from `input_gme_path` and will be saved into `workdir_path`.

    Note that this step might fail on some input GME files (which contain data after the audio table). See my blog post and [this link](https://github.com/entropia/tip-toi-reveng/blob/90004b5ff6239d0635cf6029654374002dc034bd/Audio/README.md) for more information.

    Args:
        workdir_path: Path to the working directory, where the updated GME will be saved into
        filelist_path: Path to filelist with updated audio files
        input_gme_path: Input GME file which

    Returns:
        Path to the new GME file
    """
    file_name = os.path.basename(input_gme_path)
    file_name_base, _ = os.path.splitext(file_name)

    new_path = os.path.join(workdir_path, f"{file_name_base} (eng).gme")

    cmd = ["./libtiptoi", "r", filelist_path, new_path, input_gme_path]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        # libtiptoi returns 1 upon success
        if exc.returncode == 1:
            pass

    return new_path


if __name__ == "__main__":
    main()
