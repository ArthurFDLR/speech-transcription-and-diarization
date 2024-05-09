# Speech Transcription and Diarization

💬 YoutubeCC - Parse JSON3 Youtube Closed Captions


```python
import dotenv, json
from pathlib import Path
from stad import WhisperPipeline, DiarizationPipeline, assign_speaker_to_transcript

# Load HF_TOKEN from .env file
dotenv.load_dotenv()

PODCAST_AUDIO_PATH = Path.cwd() / ".data" / "audio.wav"
```

## Automatic Speech Recognition with Hugging Face Transformers implementation of Whisper


```python
with WhisperPipeline.create(
    return_timestamps="word",
    attn_implementation="sdpa",
    batch_size=16,
) as whisper:
    transcript_df = whisper(
        audio_path=PODCAST_AUDIO_PATH,
        language="english",
    )

transcript_df
```

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word. Also make sure WhisperTimeStampLogitsProcessor was used during generation.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>start</th>
      <th>end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What's</td>
      <td>0.04</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>up</td>
      <td>0.30</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>everybody</td>
      <td>0.44</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>welcome</td>
      <td>0.70</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to</td>
      <td>1.00</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25428</th>
      <td>We'll</td>
      <td>7396.76</td>
      <td>7397.04</td>
    </tr>
    <tr>
      <th>25429</th>
      <td>see</td>
      <td>7397.04</td>
      <td>7397.16</td>
    </tr>
    <tr>
      <th>25430</th>
      <td>you</td>
      <td>7397.16</td>
      <td>7397.30</td>
    </tr>
    <tr>
      <th>25431</th>
      <td>there.</td>
      <td>7397.30</td>
      <td>7397.50</td>
    </tr>
    <tr>
      <th>25432</th>
      <td>Peace.</td>
      <td>7397.50</td>
      <td>7398.32</td>
    </tr>
  </tbody>
</table>
<p>25433 rows × 3 columns</p>
</div>



## Speaker Diarization with PyAnnote [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)


```python
with DiarizationPipeline() as diarization:
    speakers_df = diarization(audio_path=PODCAST_AUDIO_PATH)

speakers_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>speaker</th>
      <th>start</th>
      <th>end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>SPEAKER_03</td>
      <td>0.132219</td>
      <td>46.319094</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>SPEAKER_02</td>
      <td>24.870969</td>
      <td>25.124094</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>SPEAKER_03</td>
      <td>47.567844</td>
      <td>58.705344</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>SPEAKER_04</td>
      <td>56.376594</td>
      <td>57.540969</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E</td>
      <td>SPEAKER_03</td>
      <td>59.430969</td>
      <td>64.206594</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3208</th>
      <td>DSK</td>
      <td>SPEAKER_03</td>
      <td>7388.597844</td>
      <td>7389.272844</td>
    </tr>
    <tr>
      <th>3209</th>
      <td>DSL</td>
      <td>SPEAKER_03</td>
      <td>7390.049094</td>
      <td>7391.669094</td>
    </tr>
    <tr>
      <th>3210</th>
      <td>DSM</td>
      <td>SPEAKER_00</td>
      <td>7391.669094</td>
      <td>7391.702844</td>
    </tr>
    <tr>
      <th>3211</th>
      <td>DSN</td>
      <td>SPEAKER_00</td>
      <td>7395.145344</td>
      <td>7397.541594</td>
    </tr>
    <tr>
      <th>3212</th>
      <td>DSO</td>
      <td>SPEAKER_00</td>
      <td>7397.963469</td>
      <td>7398.300969</td>
    </tr>
  </tbody>
</table>
<p>3213 rows × 4 columns</p>
</div>



## Assign speaker labels to each chunk in the transcript


```python
assign_speaker_to_transcript(
    speakers_df=speakers_df,
    transcript_df=transcript_df,
    inplace=True,
)

transcript_df.to_json(
    PODCAST_AUDIO_PATH.with_suffix(".json"), orient="records", indent=4
)

transcript_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>start</th>
      <th>end</th>
      <th>speaker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What's</td>
      <td>0.04</td>
      <td>0.30</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>up</td>
      <td>0.30</td>
      <td>0.44</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>everybody</td>
      <td>0.44</td>
      <td>0.70</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>welcome</td>
      <td>0.70</td>
      <td>1.00</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to</td>
      <td>1.00</td>
      <td>1.22</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25428</th>
      <td>We'll</td>
      <td>7396.76</td>
      <td>7397.04</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>25429</th>
      <td>see</td>
      <td>7397.04</td>
      <td>7397.16</td>
      <td>SPEAKER_00</td>
    </tr>
    <tr>
      <th>25430</th>
      <td>you</td>
      <td>7397.16</td>
      <td>7397.30</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>25431</th>
      <td>there.</td>
      <td>7397.30</td>
      <td>7397.50</td>
      <td>SPEAKER_00</td>
    </tr>
    <tr>
      <th>25432</th>
      <td>Peace.</td>
      <td>7397.50</td>
      <td>7398.32</td>
      <td>SPEAKER_03</td>
    </tr>
  </tbody>
</table>
<p>25433 rows × 4 columns</p>
</div>




```python

```
