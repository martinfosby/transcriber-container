import os
from pathlib import Path
from typing import Dict, Any
import datetime
import time
import json
import asyncio
from transformers import pipeline, Pipeline
import torch

# User defined imports
from ContainerName import ContainerName
from BlobStorageService import BlobStorageService
from AsyncConfigManager import AsyncConfigManager
from AudioProcessor import AudioProcessor

from logger_config import get_logger, setup_logging
import logging
setup_logging(logging.INFO)
logger = get_logger(__name__)

class AsyncWhisperTranscriber:
    """Main class for transcribing audio using Whisper models with async support."""
    
    def __init__(self, 
                 model_path: str = "NbAiLabBeta/nb-whisper-large-verbatim",
                 ):
        """Initialize the transcriber.
        
        Args:
            model_path: Path or name of the transcription model to use
            download_folder: Local folder for downloading audio files
        """
        logger.info("Initializing AsyncWhisperTranscriber")
        logger.info(f"Model path: {model_path}")
        self.model_path = model_path
        self.timestamp = datetime.datetime.now()
        
        # Initialize configuration manager
        logger.info("Initializing configuration manager")
        self.config = AsyncConfigManager.get_instance()
        self.blob_storage_service = BlobStorageService(self.config)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Ensure folders exist
        os.makedirs("transcriptions", exist_ok=True)
        
        # These will be initialized during transcribe
        self.asr = None
        self.local_audio_path = None
        self.basename = None
        self.filename = None
        self.file_extension = None
        self.modelname = None
        self.result_filename = None

    async def initialize(self):
        """Initialize async resources."""
        logger.info("Initializing async resources...")

        # Validate ASR model
        # logger.info("Validating ASR model...")
        # self.validate_asr_model_task = asyncio.create_task(self.__validate_asr_model(self.model_path))
        logger.info(f"Loading ASR model: {self.model_path}")
        # Load the ASR model (in a thread pool as it's CPU-bound)
        self.load_model_task = asyncio.create_task(self.load_model())


        if self.config.args.use_call_recording:
            logger.info("Using call recording...")
            logger.info("Downloading recording call data...")
            download_recording_call_data_task = asyncio.create_task(self.blob_storage_service.download_blob_from_container(ContainerName.RECORDINGS_CALL_DATA, self.config.blob_name))
            self.recording_call_data_file = await download_recording_call_data_task
            await self.load_and_process_recording()
        elif self.config.args.run_webapp:
            logger.info("Using telephone json data...")
            self.recording_call_data = self.config.telephone_json_data
            await self.load_and_process_recording()



    async def load_and_process_recording(self):
        if not self.recording_call_data:
            # Load the recording call data from file
            with open(self.recording_call_data_file, "rb") as f:
                self.recording_call_data = json.load(f)

            self.config.json_data_from_telephone = self.recording_call_data.get("json_data_from_telephone", False)

        logger.info("Getting call recording and metadata...")
        self.recording_and_metadata = await self.blob_storage_service.get_call_recording_and_metadata(
                self.recording_call_data
        )

        if self.config.json_data_from_telephone:
            logger.info("Sent from telephone system...")
            self._process_audio_metadata()


    def _process_audio_metadata(self):
        recording_info = self.recording_and_metadata.get("recordingStorageInfo", {})
        content_type = recording_info.get("contentType")
        channel_type = recording_info.get("channelType")
        format_ = recording_info.get("format")
        audio_config = recording_info.get("audioConfiguration", {})
        sample_rate = audio_config.get("sampleRate")
        bit_rate = audio_config.get("bitRate")
        channels = audio_config.get("channels")

        logger.info(f"Content type: {content_type}")
        logger.info(f"Channel type: {channel_type}")
        logger.info(f"Format: {format_}")
        logger.info(f"Sample rate: {sample_rate}")
        logger.info(f"Bit rate: {bit_rate}")
        logger.info(f"Channels: {channels}")
        logger.info(f"Checking content type and channel type...")

        if content_type == "audio" and channel_type == "mixed":
            if format_ != "mp3":
                logger.info(f"Converting format to 'mp3'...")
                self.audio = self.audio_processor.convert_audio_to_mp3(self.audio)
            elif bit_rate is None or bit_rate > 35000:
                logger.info(f"Setting bitrate to 25000...")
                self.audio = self.audio_processor.convert_audio_to_mp3(self.audio)
            elif sample_rate != 16000:
                logger.info(f"Converting sample rate to 16000...")
                self.audio = self.audio_processor.convert_audio_to_mp3(self.audio)

        elif content_type == "audio" and channel_type == "unmixed":
            if format_ != "wav":
                logger.info(f"Converting format to 'wav'...")
                self.audio = self.audio_processor.convert_audio_to_wav(self.audio)
            elif sample_rate != 16000:
                logger.info(f"Converting sample rate to 16000...")
                self.audio = self.audio_processor.convert_audio_to_wav(self.audio)



    async def load_model(self):
        """Load the ASR model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading ASR model on {device}")
        self.asr: Pipeline = await asyncio.to_thread(
            pipeline,
            "automatic-speech-recognition",
            model=self.model_path,
            device=device
        )

    @staticmethod
    async def __validate_asr_model(model_path: str) -> None:
        """Validates that the model is suitable for speech recognition."""
        try:
            from huggingface_hub import model_info
            from transformers import AutoConfig
            # Run blocking functions in separate threads using asyncio.to_thread
            try:
                info = await asyncio.to_thread(model_info, model_path)

                tags = info.tags
                asr_indicators = ["automatic-speech-recognition", "asr", "whisper", "wav2vec", "speech-recognition"]

                if any(tag.lower() in asr_indicators for tag in tags):
                    logger.info(f"Model {model_path} validated as ASR model via HF Hub tags")
                    return

                if "pipeline_tag" in info.cardData:
                    if info.cardData["pipeline_tag"] == "automatic-speech-recognition":
                        logger.info(f"Model {model_path} validated as ASR model via pipeline tag")
                        return

            except Exception as e:
                logger.warning(f"Could not validate model via Hugging Face Hub: {str(e)}")

            try:
                config = await asyncio.to_thread(AutoConfig.from_pretrained, model_path)
                model_type = getattr(config, "model_type", "").lower()
                asr_architectures = ["whisper", "wav2vec2", "hubert", "wavlm", "unispeech"]

                if any(arch in model_type for arch in asr_architectures):
                    logger.info(f"Model {model_path} validated as ASR model via config (architecture: {model_type})")
                    return

            except Exception as e:
                logger.warning(f"Could not validate model via config: {str(e)}")

            common_asr_prefixes = ["openai/whisper", "facebook/wav2vec", "NbAiLab/nb-whisper"]
            if any(model_path.startswith(prefix) for prefix in common_asr_prefixes):
                logger.info(f"Model {model_path} assumed to be ASR model based on naming convention")
                return

            logger.warning(f"Could not validate that {model_path} is an ASR model. Proceeding with caution.")

        except ImportError:
            logger.warning("huggingface_hub not installed; skipping model validation")
            return

        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            logger.warning("Proceeding without model validation")


    async def transcribe(self) -> Dict[str, Any]:
        """Transcribe the audio file and return the results."""
        logger.info("Transcribing audio...")
        try:
            # Initialize resources
            await self.initialize()
            
            await self.load_model_task  # Ensure model is loaded

            # Track transcription time
            logger.info("Measuring transcription time...")
            start_time = time.time()
            
            # Perform transcription (run in thread pool as it's CPU-bound)
            logger.info("Starting transcription")
            transcription = self.asr(
                self.recording_and_metadata['content'],
                chunk_length_s=self.config.args.chunk_size,
                return_timestamps=self.config.args.return_timestamps,
                generate_kwargs={
                    'num_beams': self.config.args.num_beams, 
                    'task': self.config.args.task, 
                    'language': self.config.args.language
                }
            )
            
            # Calculate duration
            end_time = time.time()
            transcription_time = end_time - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            
            # # Get audio duration (run in thread pool)
            # waveform, sr = await asyncio.to_thread(torchaudio.load(self.local_audio_path))
            # audio_duration = waveform.shape[1] / sr
            
            # Prepare results
            logger.info("Preparing transcription results...")
            self.transcription_result = {
                "model": self.model_path,
                "cointainer_env": self.config.container_env,
                "timestamp": self.timestamp.isoformat(),
                "transcription": transcription['text'],
                "transcription_chunks": transcription.get('chunks'),
                "transcription_time_taken": transcription_time,
                "recording_metadata": self.recording_and_metadata['metadata'],
                "call_metadata": self.recording_call_data
            }
            logger.info(f"Transcription results prepared {self.transcription_result}")

            return self.transcription_result
            
            
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise


    async def save_results(self, result: Dict[str, Any]) -> str:
        """Save transcription results to a local JSON file asynchronously."""
        logger.info("Saving transcription results...")
        try:
            tones = self.transcription_result.get('call_metadata', {}).get('tones_interpreted', 'unknown')
            filename = f"{tones}-{self.timestamp.strftime('%Y-%m-%d-%H-%M-%S')}.json"
            result_dir = Path("transcriptions")
            result_dir.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

            result_path = result_dir / filename
            logger.info(f"Saving transcription results to {result_path}")

            json_str = json.dumps(result, ensure_ascii=False, indent=4)
            await asyncio.to_thread(result_path.write_text, data=json_str, encoding="utf-8")
            
            return str(result_path)

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise ValueError("Failed to save transcription results") from e

