"""Speech to speech translation using Meta's seamless_communication, expressive model.

Repo: https://github.com/facebookresearch/seamless_communication
Inference is based on: https://github.com/facebookresearch/seamless_communication/blob/90e2b57ac4d82fa2bfaa25caeffe39ceb8b2ebec/src/seamless_communication/cli/expressivity/predict/predict.py
"""

import os
import torch
import argparse
import torchaudio
from pathlib import Path
from typing import NamedTuple
from fairseq2.data import SequenceData
from seamless_communication.inference import Translator
from fairseq2.data.audio import WaveformToFbankConverter
from seamless_communication.store import add_gated_assets
from seamless_communication.models.unity import load_gcmvn_stats, load_unity_unit_tokenizer
from seamless_communication.cli.m4t.predict import set_generation_opts, add_inference_arguments
from seamless_communication.cli.expressivity.predict.pretssel_generator import PretsselGenerator
from seamless_communication.cli.expressivity.predict.predict import remove_prosody_tokens_from_text


class TranslatedAudio(NamedTuple):
    """Hold results of a translated audio. This is intended to hold data returned by `translate_audio_files()`."""
    path: str
    text: str


def translate_audio_files(input_paths: list[str],  # pylint: disable=too-many-arguments, too-many-locals
                          output_directory: str,
                          *,
                          target_language: str = "eng",
                          model_name: str = "seamless_expressivity",
                          vocoder_name: str = "vocoder_pretssel",
                          duration_factor: float = 1.0,
                          force_cpu: bool = False) -> list[TranslatedAudio]:
    """Translate audio files specified in `input_paths`, saving them into the `output_directory`.

    Args:
        input_paths: Files to be translated
        output_directory: Destination of translated audio, audio files will be in mp3 format
        force_cpu: Force CPU inference even if a GPU is available (but when it doesn't have enough memory)

    Returns:
        Tuple of the output path and transcript
    """
    # Inference setup
    add_gated_assets(Path("SeamlessExpressive"))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    if force_cpu:  # For tests
        device = torch.device("cpu")
        dtype = torch.float32

    unit_tokenizer = load_unity_unit_tokenizer(model_name)

    translator = Translator(
        model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype
    )

    pretssel_generator = PretsselGenerator(
        vocoder_name,
        vocab_info=unit_tokenizer.vocab_info,
        device=device,
        dtype=dtype
    )

    fbank_extractor = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=False,
        device=device,
        dtype=dtype
    )

    _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(vocoder_name)
    gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
    gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

    # Args hacking
    parser = argparse.ArgumentParser(description="Running SeamlessExpressive inference.")
    parser = add_inference_arguments(parser)
    args = parser.parse_args([])
    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    # Output directory
    os.makedirs(output_directory, exist_ok=True)

    # List which will be returned
    out = []

    # Per-audio file inference
    for path in input_paths:
        wav, sample_rate = torchaudio.load(path)
        wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16_000)
        wav = wav.transpose(0, 1)

        data = fbank_extractor(
            {
                "waveform": wav,
                "sample_rate": 16_000,
            }
        )

        fbank = data["fbank"]
        gcmvn_fbank = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
        std, mean = torch.std_mean(fbank, dim=0)
        fbank = fbank.subtract(mean).divide(std)

        src = SequenceData(
            seqs=fbank.unsqueeze(0),
            seq_lens=torch.LongTensor([fbank.shape[0]]),
            is_ragged=False,
        )
        src_gcmvn = SequenceData(
            seqs=gcmvn_fbank.unsqueeze(0),
            seq_lens=torch.LongTensor([gcmvn_fbank.shape[0]]),
            is_ragged=False,
        )

        text_output, unit_output = translator.predict(
            src,
            "s2st",
            target_language,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
            duration_factor=duration_factor,
            prosody_encoder_input=src_gcmvn,
        )

        assert unit_output is not None
        speech_output = pretssel_generator.predict(
            unit_output.units,
            tgt_lang=target_language,
            prosody_encoder_input=src_gcmvn,
        )

        # File name
        file_name = os.path.basename(path)
        file_name_base, _ = os.path.splitext(file_name)
        output_path = os.path.join(output_directory, f"{file_name_base}.mp3")

        # Save
        torchaudio.save(
            output_path,
            speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
            sample_rate=speech_output.sample_rate,
        )

        # Print transcript
        text_out = remove_prosody_tokens_from_text(str(text_output[0]))

        out.append(TranslatedAudio(output_path, text_out))

    return out
