from pathlib import Path
from typing import Optional, Tuple

import ffmpeg
import numpy as np


def probe_audio(file_path: Path):
    """
    Probes an audio file to get its properties such as sample rate and number of channels.

    Args:
        file_path (Path): The path to the audio file.

    Returns:
        tuple: A tuple containing the sample rate (int) and the number of channels (int) of the audio file.

    Raises:
        ffmpeg.Error: If an error occurs during probing, the error details are printed and the exception is raised.
    """
    try:
        probe_info = ffmpeg.probe(str(file_path), select_streams="a")
        audio_info = probe_info["streams"][0]
        sample_rate = int(audio_info["sample_rate"])
        channels = int(audio_info["channels"])
        return sample_rate, channels
    except ffmpeg.Error as e:
        print("An error occurred during probing:", e.stderr.decode())
        raise


def read_audio(
    file_path: Path,
    sample_rate_target: Optional[int] = None,
    channels_target: Optional[int] = None,
    dtype_target: np.dtype = np.int16,
) -> Tuple[np.ndarray, int]:
    """
    Reads an audio file and returns its samples as a NumPy array with optional specifications for sample rate and channels.

    This function first probes the audio file to determine its native sample rate and channel count, unless
    both are explicitly provided. It then uses FFmpeg to convert the audio file to raw audio data which is
    subsequently converted to a NumPy array.

    Args:
        file_path (Path): The path to the audio file.
        sample_rate_target (Optional[int]): The target sample rate to which the audio should be converted.
                                            If None, the native sample rate of the file is used.
        channels_target (Optional[int]): The target number of channels for the audio. If None, the native
                                         number of channels of the file is used.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the audio samples as a NumPy array and the sample rate of the audio.

    Raises:
        ffmpeg.Error: If an error occurs during the audio conversion, the error details are captured and
                      an exception is raised.
    """

    # Probe the file to get audio properties
    if sample_rate_target is None or channels_target is None:
        sample_rate_probe, channels_probe = probe_audio(file_path)

    sample_rate = sample_rate_target or sample_rate_probe
    channels = channels_target or channels_probe

    data_format_map = {
        np.int16: ("s16le", "pcm_s16le"),
        np.float32: ("f32le", "pcm_f32le"),
    }
    if dtype_target not in data_format_map.keys():
        raise (
            f"Unsupported data type ({dtype_target}). Please use one of {list(data_format_map.keys())}."
        )
    output_format, codec = data_format_map[dtype_target]

    # Set up an FFmpeg process that outputs raw audio data
    out, _ = (
        ffmpeg.input(str(file_path))
        .output(
            "pipe:",
            format=output_format,
            acodec=codec,
            ac=channels,
            ar=sample_rate,
        )
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Convert the raw audio bytes to a NumPy array
    audio_array = np.frombuffer(out, dtype_target)

    # Reshape the array into multiple channels if necessary
    if channels > 1:
        audio_array = np.reshape(audio_array, (-1, channels))

    return audio_array, sample_rate
