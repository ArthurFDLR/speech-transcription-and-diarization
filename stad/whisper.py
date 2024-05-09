import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


@dataclass
class ModelConfig:
    """
    Configuration settings for the model, including parameters that influence memory and performance.

    Attributes:
        pretrained_model_name_or_path (str): The name or path for the pretrained model.
        torch_dtype (torch.dtype): Data type for model tensors, e.g., torch.float16, torch.float32.
        attn_implementation (Literal["flash_attention_2", "sdpa", "eager"]): The attention mechanism implementation.
        low_cpu_mem_usage (bool): Flag to optimize model for low CPU memory usage.
        use_safetensors (bool): Enables SafeTensors to manage memory more effectively, useful in constrained environments.
    """

    pretrained_model_name_or_path: str
    torch_dtype: torch.dtype
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"]
    low_cpu_mem_usage: bool
    use_safetensors: bool


@dataclass
class PipelineConfig:
    """
    Configuration settings for the pipeline, defining how the processing is handled.

    Attributes:
        task (str): The task type, e.g., "automatic-speech-recognition".
        max_new_tokens (int): Maximum number of new tokens to generate in the output.
        chunk_length_s (int): Length of audio chunks in seconds.
        batch_size (int): The size of the batch processing.
        return_timestamps (Union[bool, Literal["word"]]): Specifies if timestamps should be returned, and at what granularity.
        device (torch.device): The device (CPU/GPU) on which to run the computations.
    """

    task: str
    max_new_tokens: int
    chunk_length_s: int
    batch_size: int
    return_timestamps: Union[bool, Literal["word"]]
    device: torch.device


class WhisperPipeline:
    """
    A pipeline for processing speech recognition tasks using the Whisper model.

    This class provides an interface to load and execute the Whisper model, handling the configuration and data processing.
    It is designed to be used as a context manager, ensuring that resources are properly managed.

    Attributes:
        model_config (ModelConfig): Configuration for the model specifics.
        pipeline_config (PipelineConfig): Configuration for the pipeline execution.

    Example:
        ```python
        with WhisperPipeline.create() as whisper:
            output = whisper(audio_path="path/to/audio.wav")
        ```
    """

    def __init__(self, model_config: ModelConfig, pipeline_config: PipelineConfig):
        """
        Initializes the WhisperPipeline with model and pipeline configurations.
        Use the factory method (`create`) to create an instance with custom configurations.

        Args:
            model_config (ModelConfig): Configuration for the model specifics.
            pipeline_config (PipelineConfig): Configuration for the pipeline execution.
        """
        self._model_config = model_config
        self._pipeline_config = pipeline_config

        self._pipeline = None
        self._model = None
        self._processor = None

    @property
    def model_config(self) -> ModelConfig:
        return asdict(self._model_config)

    @property
    def pipeline_config(self) -> PipelineConfig:
        return asdict(self._pipeline_config)

    @property
    def config(self):
        """
        Returns the configuration settings for the model and pipeline.

        Returns:
            dict: A dictionary containing the model and pipeline configurations.
        """
        return {
            "model_config": self.model_config,
            "pipeline_config": self.pipeline_config,
        }

    @classmethod
    def create(
        cls,
        model_id: str = "openai/whisper-large-v3",
        torch_dtype: torch.dtype = torch.float16,
        device: Optional[Union[torch.device, str]] = None,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "eager",
        batch_size: int = 16,
        return_timestamps: Union[bool, Literal["word"]] = False,
        low_cpu_mem_usage: bool = True,
        use_safetensors: bool = True,
        max_new_tokens: int = 128,
        chunk_length_s: int = 30,
        task: str = "automatic-speech-recognition",
    ):
        """
        Factory method to create a new instance of WhisperPipeline with default or specified configurations.

        Args:
            model_id (str): The identifier for the pretrained model. Defaults to "openai/whisper-large-v3".
            torch_dtype (torch.dtype): Data type for model tensors. Defaults to torch.float16.
            device (Optional[Union[torch.device, str]]): Computation device ('cpu' or 'cuda:0'). Defaults to GPU if available.
            attn_implementation (Literal["flash_attention_2", "sdpa", "eager"]): The attention mechanism implementation. Flash Attention 2 is recommended for better performance, but requires isn't supported for word timestamps. Defaults to "eager".
            batch_size (int): Number of samples per batch. Defaults to 16.
            return_timestamps (Union[bool, Literal["word"]]): Whether to include timestamps in the output. Defaults to False.
            low_cpu_mem_usage (bool): Optimize for lower CPU memory usage. Defaults to True.
            use_safetensors (bool): Use SafeTensors for better memory management. Defaults to True.
            max_new_tokens (int): Maximum tokens for the output. Defaults to 128.
            chunk_length_s (int): Length of audio chunks to process at a time. Defaults to 30.
            task (str): Type of task, e.g., "automatic-speech-recognition". Defaults to "automatic-speech-recognition".

        Returns:
            WhisperPipeline: An instance of the WhisperPipeline class.
        """

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        if isinstance(device, str):
            device = torch.device(device)

        return cls(
            model_config=ModelConfig(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_safetensors=use_safetensors,
            ),
            pipeline_config=PipelineConfig(
                task=task,
                max_new_tokens=max_new_tokens,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                return_timestamps=return_timestamps,
                device=device,
            ),
        )

    def _load_pipeline(self):
        """
        Loads the model and processor required for the pipeline based on the configuration.
        """
        self._processor = AutoProcessor.from_pretrained(
            self._model_config.pretrained_model_name_or_path
        )

        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            **asdict(self._model_config)
        )
        self._model.to(self._pipeline_config.device)

        self._pipeline = pipeline(
            **asdict(self._pipeline_config),
            model=self._model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
        )

    def __enter__(self):
        """
        Context management entry. Ensures that the pipeline is loaded when the context is entered.

        Returns:
            WhisperPipeline: The instance of the pipeline.
        """
        self._load_pipeline()
        return self

    def __call__(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Processes an audio file using the loaded Whisper model and returns the transcription.

        Args:
            audio_path (str): Path to the audio file to be processed.
            language (Optional[str]): If specified, processes the audio in the given language. Defaults to None.

        Returns:
            pd.DataFrame: Transcription output as a DataFrame containing the chunks of text.

        Raises:
            RuntimeError: If the pipeline is not loaded or the output is not as expected.
        """

        if self._pipeline is None:
            raise RuntimeError(
                "Pipeline not loaded. Use `with` statement to load pipeline."
            )

        generate_kwargs = {"language": language} if language else {}
        output = self._pipeline(str(audio_path), generate_kwargs=generate_kwargs)
        if output is None or not isinstance(output, dict) or "chunks" not in output:
            raise RuntimeError("Unexpected output from pipeline.")

        df = pd.DataFrame(output["chunks"])
        df[["start", "end"]] = pd.DataFrame(df.timestamp.tolist())
        df.drop(columns="timestamp", inplace=True)
        return df

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context management exit. Clears the pipeline and ensures proper cleanup to free memory.

        Args:
            exc_type: The type of the exception.
            exc_val: The value of the exception.
            exc_tb: The traceback of the exception.
        """

        self._clear_pipeline()

    def _clear_pipeline(self):
        """
        Clears the loaded model, processor, and pipeline to free up resources, particularly GPU memory.
        """
        if self._pipeline:
            del self._pipeline
        if self._model:
            del self._model
        if self._processor:
            del self._processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._pipeline = None
        self._model = None
        self._processor = None
