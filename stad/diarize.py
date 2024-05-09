import gc
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline

from .audio import read_audio


class DiarizationPipeline:
    _SAMPLE_RATE = 16000
    _CHANNELS = 1
    _AUDIO_DTYPE = np.float32

    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initializes the DiarizationPipeline with specified model, token, and computation device.

        Args:
            model_name (str): Identifier for the model to be used from Hugging Face. Defaults to 'pyannote/speaker-diarization-3.1'.
            hf_token (Optional[str]): Authentication token for Hugging Face for gated models ('pyannote/speaker-diarization-3.1'). Defaults to None.
            device (Optional[Union[torch.device, str]]): The device (e.g., 'cpu' or 'cuda') on which to perform computations. Defaults to available GPU, or CPU if GPU is not available.
        """
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        if isinstance(device, str):
            device = torch.device(device)

        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN", None)

        self._device = device
        self._model_name = model_name
        self._hf_token = hf_token

        self._model = None

    def _load_model(self):
        """Loads the diarization model from Hugging Face."""

        self._model = Pipeline.from_pretrained(
            self._model_name,
            use_auth_token=self._hf_token,
        ).to(self._device)

    def _clear_model(self):
        """Clears the model from memory, managing GPU and general resources."""

        if self._model:
            del self._model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model = None

    def __enter__(self):
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._clear_model()

    def __call__(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Processes the audio file for speaker diarization.

        Args:
            audio_path (Union[str, Path]): Path to the audio file.
            num_speakers (Optional[int]): Expected number of speakers, if known. Defaults to None.
            min_speakers (Optional[int]): Minimum number of speakers to detect. Defaults to None.
            max_speakers (Optional[int]): Maximum number of speakers to detect. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with columns 'label', 'speaker', 'start', and 'end' denoting the segments of detected speakers.
        """
        audio, _ = read_audio(
            audio_path,
            sample_rate_target=self._SAMPLE_RATE,
            channels_target=self._CHANNELS,
            dtype_target=self._AUDIO_DTYPE,
        )

        audio = torch.from_numpy(audio)

        segments = self._model(
            dict(
                waveform=audio[None, :],
                sample_rate=self._SAMPLE_RATE,
            ),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
        diarize_df.drop(columns=["segment"], inplace=True)
        return diarize_df
        return diarize_df
