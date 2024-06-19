# Speech Transcription and Diarization

💬 YoutubeCC - Parse JSON3 Youtube Closed Captions


```python
import dotenv, json
from pathlib import Path
from stad import WhisperPipeline, DiarizationPipeline, assign_speaker_to_transcript

# Load HF_TOKEN from .env file
dotenv.load_dotenv()

AUDIO_PATH = Path.cwd() / ".data" / "audio.wav"
```

    /home/arthur/Documents/02.workspace/02.active/speech-transcription-and-diarization/venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


## Automatic Speech Recognition with Hugging Face Transformers implementation of Whisper


```python
with WhisperPipeline.create(
    return_timestamps="word",
    attn_implementation="sdpa",
    batch_size=16,
) as whisper:
    transcript_df = whisper(
        audio_path=AUDIO_PATH,
        language="english",
    )

transcript_df
```

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    WhisperModel is using WhisperSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.
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
      <td>0 days 00:00:00.040000</td>
      <td>0 days 00:00:00.300000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>up</td>
      <td>0 days 00:00:00.300000</td>
      <td>0 days 00:00:00.440000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>everybody</td>
      <td>0 days 00:00:00.440000</td>
      <td>0 days 00:00:00.700000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>welcome</td>
      <td>0 days 00:00:00.700000</td>
      <td>0 days 00:00:01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to</td>
      <td>0 days 00:00:01</td>
      <td>0 days 00:00:01.220000</td>
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
      <td>0 days 02:03:16.760000</td>
      <td>0 days 02:03:17.040000</td>
    </tr>
    <tr>
      <th>25429</th>
      <td>see</td>
      <td>0 days 02:03:17.040000</td>
      <td>0 days 02:03:17.160000</td>
    </tr>
    <tr>
      <th>25430</th>
      <td>you</td>
      <td>0 days 02:03:17.160000</td>
      <td>0 days 02:03:17.300000</td>
    </tr>
    <tr>
      <th>25431</th>
      <td>there.</td>
      <td>0 days 02:03:17.300000</td>
      <td>0 days 02:03:17.500000</td>
    </tr>
    <tr>
      <th>25432</th>
      <td>Peace.</td>
      <td>0 days 02:03:17.500000</td>
      <td>0 days 02:03:18.320000</td>
    </tr>
  </tbody>
</table>
<p>25433 rows × 3 columns</p>
</div>



## Speaker Diarization with PyAnnote [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)


```python
with DiarizationPipeline() as diarization:
    speakers_df = diarization(audio_path=AUDIO_PATH)

speakers_df
```

    torchvision is not available - cannot save figures





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
      <td>0 days 00:00:00.132218750</td>
      <td>0 days 00:00:46.319093750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>SPEAKER_02</td>
      <td>0 days 00:00:24.870968750</td>
      <td>0 days 00:00:25.124093750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>SPEAKER_03</td>
      <td>0 days 00:00:47.567843750</td>
      <td>0 days 00:00:58.705343750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>SPEAKER_04</td>
      <td>0 days 00:00:56.376593750</td>
      <td>0 days 00:00:57.540968750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E</td>
      <td>SPEAKER_03</td>
      <td>0 days 00:00:59.430968750</td>
      <td>0 days 00:01:04.206593750</td>
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
      <td>0 days 02:03:08.597843750</td>
      <td>0 days 02:03:09.272843750</td>
    </tr>
    <tr>
      <th>3209</th>
      <td>DSL</td>
      <td>SPEAKER_03</td>
      <td>0 days 02:03:10.049093750</td>
      <td>0 days 02:03:11.669093750</td>
    </tr>
    <tr>
      <th>3210</th>
      <td>DSM</td>
      <td>SPEAKER_00</td>
      <td>0 days 02:03:11.669093750</td>
      <td>0 days 02:03:11.702843750</td>
    </tr>
    <tr>
      <th>3211</th>
      <td>DSN</td>
      <td>SPEAKER_00</td>
      <td>0 days 02:03:15.145343750</td>
      <td>0 days 02:03:17.541593750</td>
    </tr>
    <tr>
      <th>3212</th>
      <td>DSO</td>
      <td>SPEAKER_00</td>
      <td>0 days 02:03:17.963468750</td>
      <td>0 days 02:03:18.300968750</td>
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
    AUDIO_PATH.with_suffix(".json"), orient="records", indent=4
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
      <td>0 days 00:00:00.040000</td>
      <td>0 days 00:00:00.300000</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>up</td>
      <td>0 days 00:00:00.300000</td>
      <td>0 days 00:00:00.440000</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>everybody</td>
      <td>0 days 00:00:00.440000</td>
      <td>0 days 00:00:00.700000</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>welcome</td>
      <td>0 days 00:00:00.700000</td>
      <td>0 days 00:00:01</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to</td>
      <td>0 days 00:00:01</td>
      <td>0 days 00:00:01.220000</td>
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
      <td>0 days 02:03:16.760000</td>
      <td>0 days 02:03:17.040000</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>25429</th>
      <td>see</td>
      <td>0 days 02:03:17.040000</td>
      <td>0 days 02:03:17.160000</td>
      <td>SPEAKER_00</td>
    </tr>
    <tr>
      <th>25430</th>
      <td>you</td>
      <td>0 days 02:03:17.160000</td>
      <td>0 days 02:03:17.300000</td>
      <td>SPEAKER_03</td>
    </tr>
    <tr>
      <th>25431</th>
      <td>there.</td>
      <td>0 days 02:03:17.300000</td>
      <td>0 days 02:03:17.500000</td>
      <td>SPEAKER_00</td>
    </tr>
    <tr>
      <th>25432</th>
      <td>Peace.</td>
      <td>0 days 02:03:17.500000</td>
      <td>0 days 02:03:18.320000</td>
      <td>SPEAKER_03</td>
    </tr>
  </tbody>
</table>
<p>25433 rows × 4 columns</p>
</div>




```python

```
