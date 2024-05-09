from typing import Optional

import numpy as np
import pandas as pd


def assign_speaker_to_transcript(
    speakers_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    inplace: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Assigns speaker labels to segments of a transcript based on the closest matching end timestamps.

    This function aligns speaker information from a diarization process (speakers_df) with a transcript
    generated from automatic speech recognition (ASR), based on timestamps. Each transcript segment is
    labeled with the speaker who is identified as speaking at the end of the segment. If 'inplace' is set
    to True, the speaker labels are assigned directly to the transcript_df. Otherwise, a new DataFrame
    is returned.

    Args:
        speakers_df (pd.DataFrame): A DataFrame containing columns ['end', 'speaker'], where 'end' is
            the timestamp at which a speaker stops speaking and 'speaker' is the identifier or name of
            the speaker.
        transcript_df (pd.DataFrame): A DataFrame containing at least a column ['end'], where 'end' is
            the timestamp at which a transcript segment ends.
        inplace (bool): If True, modifies transcript_df to include a 'speaker' column. If False, returns
            a new DataFrame with the 'speaker' column added. Default is False.

    Returns:
        pd.DataFrame: The transcript DataFrame with a new 'speaker' column added. This DataFrame is either
        a new copy or the modified original, depending on the value of 'inplace'.
    """

    transcript_end = transcript_df.end.values

    transcript_speaker = []
    # align the diarizer timestamps and the ASR timestamps
    for _, segment in speakers_df.iterrows():
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(transcript_end - segment.end))

        # add the speaker to the list
        transcript_speaker += [segment.speaker] * (upto_idx + 1)

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript_end = transcript_end[upto_idx + 1 :]

        # if there are no more timestamps left, break
        if len(transcript_end) == 0:
            break

    transcript_speaker += [None] * (len(transcript_df) - len(transcript_speaker))

    if inplace:
        transcript_df["speaker"] = transcript_speaker
    else:
        df = transcript_df.copy()
        df["speaker"] = transcript_speaker
        return df
