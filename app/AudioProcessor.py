from pathlib import Path
from typing import Tuple, Any
import io
import asyncio
import torchaudio
import ffmpeg


# User defined imports
from logger_config import get_logger, setup_logging
import logging
setup_logging(logging.INFO)
logger = get_logger(__name__)

class AudioProcessor:
    """Handles audio processing tasks with async support."""

    @staticmethod
    async def convert_audio_to_mp3(bytes_io: io.BytesIO) -> io.BytesIO:
        """Converts audio to 16kHz, 25kbps mp3 asynchronously."""
        logger.info("Converting audio to 16kHz, 25kbps MP3")
        audio_stdout, _ = await asyncio.to_thread(
            lambda: ffmpeg.input("pipe:0", format="wav")
                .output("pipe:1", format="mp3", ar=16000, ab="25k", ac=1, acodec="libmp3lame")
                .run(input=bytes_io.getvalue(), capture_stdout=True, capture_stderr=True)
        )
        audio_buffer = io.BytesIO(audio_stdout)
        audio_buffer.seek(0)
        return audio_buffer

    @staticmethod
    async def convert_audio_to_wav(bytes_io: io.BytesIO) -> io.BytesIO:
        """Converts audio to 16kHz mono WAV asynchronously."""
        logger.info("Converting audio to 16kHz mono WAV")
        audio_stdout, _ = await asyncio.to_thread(
            lambda: ffmpeg.input("pipe:0")
                .output("pipe:1", format="wav", ar=16000, ac=1, acodec="pcm_s16le")
                .run(input=bytes_io.getvalue(), capture_stdout=True, capture_stderr=True)
        )
        audio_buffer = io.BytesIO(audio_stdout)
        audio_buffer.seek(0)
        return audio_buffer


    
    async def convert_bitrate_to_25k(bytes_io: io.BytesIO) -> io.BytesIO:
        logger.info("Converting audio to 25k bitrate")
        audio_stdout, _ = await asyncio.to_thread(
            lambda: ffmpeg.input("pipe:0", format="mp3")
                .output("pipe:1", format="mp3", acodec="libmp3lame", ab=25)
                .run(input=bytes_io.getvalue(), capture_stdout=True, capture_stderr=True)
        )
        audio_buffer = io.BytesIO(audio_stdout)
        audio_buffer.seek(0)
        return audio_buffer
    
    async def convert_sample_rate_to_16k(bytes_io: io.BytesIO) -> io.BytesIO:
        logger.info("Converting audio to 16k sample rate")
        audio_stdout, _ = await asyncio.to_thread(
            lambda: ffmpeg.input("pipe:0", format="mp3")
                .output("pipe:1", format="mp3", ar=16000)
                .run(input=bytes_io.getvalue(), capture_stdout=True, capture_stderr=True)
        )
        audio_buffer = io.BytesIO(audio_stdout)
        audio_buffer.seek(0)
        return audio_buffer


    @staticmethod
    async def convert_audio(file_path: str, target_bitrate: int = 32, target_sample_rate: int = 16000) -> io.BytesIO:
        """Converts audio to the required format for transcription asynchronously."""
        file_extension = Path(file_path).suffix.lower()
        logger.info(f"Processing audio file: {file_path} (format: {file_extension})")

        # Check if conversion is needed
        needs_conversion = False
        if file_extension != ".mp3":
            logger.info(f"Audio file is not in mp3 format, converting {file_extension} to mp3")
            needs_conversion = True
        else:
            try:
                current_bitrate, current_sample_rate = await AudioProcessor.get_audio_properties(file_path)
                logger.info(f"Current audio properties: {current_bitrate}kbps, {current_sample_rate}Hz")
                if current_bitrate != target_bitrate or current_sample_rate != target_sample_rate:
                    logger.info(f"Audio properties need adjustment")
                    needs_conversion = True
            except Exception as e:
                logger.warning(f"Could not determine audio properties: {str(e)}")
                needs_conversion = True

        # Convert if needed
        if needs_conversion:
            try:
                logger.info(f"Converting audio to {target_bitrate}k bitrate and {target_sample_rate}Hz sample rate")
                audio_stdout, _ = await asyncio.to_thread(
                    lambda: ffmpeg.input(file_path)
                        .output("pipe:1", format="mp3", audio_bitrate=f"{target_bitrate}k", ar=target_sample_rate)
                        .run(capture_stdout=True, capture_stderr=True)
                )
                audio_buffer = io.BytesIO(audio_stdout)
                audio_buffer.seek(0)
                return audio_buffer
            except Exception as e:
                logger.error(f"Audio conversion failed: {str(e)}")
                raise
        else:
            logger.info("Audio already in correct format, no conversion needed")
            data = await asyncio.to_thread(Path(file_path).read_bytes)
            audio_buffer = io.BytesIO(data)
            audio_buffer.seek(0)
            return audio_buffer

    @staticmethod
    async def get_audio_properties(file_path: str) -> Tuple[int, int]:
        """Extracts the bitrate and sample rate of an audio file using FFmpeg asynchronously."""
        logger.info(f"Extracting audio properties from {file_path}")
        info = await asyncio.to_thread(
            lambda: ffmpeg.probe(file_path, select_streams="a", show_entries="stream=bit_rate,sample_rate")
        )

        if not info.get('streams'):
            raise ValueError(f"No audio streams found in {file_path}")

        stream = info['streams'][0]
        bitrate = int(stream.get('bit_rate', 0))  # bps
        sample_rate = int(stream.get('sample_rate', 0))  # Hz

        return bitrate // 1000, sample_rate  # Convert to kbps

    @staticmethod
    async def load_audio_for_model(audio_buffer: io.BytesIO) -> Tuple[Any, int]:
        """Loads and processes audio for the ASR model asynchronously."""
        logger.info("Loading audio into memory")

        waveform, sample_rate = await asyncio.to_thread(
            lambda: torchaudio.load(audio_buffer, format="mp3")
        )

        if waveform.shape[0] > 1:
            logger.info(f"Converting audio from {waveform.shape[0]} channels to mono")
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != 16000:
            logger.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
            waveform = await asyncio.to_thread(
                lambda: torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            )
            sample_rate = 16000

        waveform_np = waveform.squeeze(0).numpy()
        return waveform_np, sample_rate