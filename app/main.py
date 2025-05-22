import os
import sys
import argparse
from typing import Dict, Any, List
import json
import asyncio
import logging

# User defined imports
from BlobStorageService import BlobStorageService
from logger_config import get_logger, setup_logging
from AsyncConfigManager import AsyncConfigManager
from AudioProcessor import AudioProcessor
from AsyncWhisperTranscriber import AsyncWhisperTranscriber

setup_logging(logging.INFO)
logger = get_logger(__name__)

async def process_multiple_files(file_paths: List[str], model_path: str) -> List[Dict[str, Any]]:
    """Process multiple audio files concurrently."""
    tasks = []
    for file_path in file_paths:
        # Prepare environment variables for each file
        os.environ["BLOB_URI"] = file_path
        
        # Create a transcriber instance
        transcriber = AsyncWhisperTranscriber(model_path=model_path)
        
        # Add to tasks
        tasks.append(transcriber.transcribe())
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing {file_paths[i]}: {str(result)}")
        else:
            final_results.append(result)
    
    return final_results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe audio files using Azure Container Instance and NB-Whisper.")

    parser.add_argument("--run-webapp", action="store_true", help="Run the Flask web application")

    parser.add_argument("--port", type=int, default=5000,
                        help="Port number to run the Flask app on")
    
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host address to run the Flask app on")
    
    parser.add_argument("--debug", action="store_true", help="Run web app in debug mode")

    parser.add_argument("--model", type=str, 
                        default="NbAiLabBeta/nb-whisper-large-verbatim",
                        help="Path to the Whisper model")

    parser.add_argument("--bitrate", type=int, default=32,
                        help="Target audio bitrate in kbps")
    
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Target audio sample rate in Hz")
    
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    
    parser.add_argument("--audio-files", type=str, nargs="+",
                        help="Multiple audio files to process")

    parser.add_argument("--audio-file", type=str,
                        help="Single audio file to process")
    
    parser.add_argument("--json-file", type=str,
                        help="Single json file to process")
    
    parser.add_argument("--use-blob-audio-url", action="store_true",
                        help="Azure blob URL for audio file")
    
    parser.add_argument("--use-blob-metadata-url", action="store_true",
                        help="Azure blob URL for metadata file")
    
    parser.add_argument("--blob-audio-url", type=str,
                        help="Provide the Azure blob URL for audio file")
    
    parser.add_argument("--blob-uri", type=str,
                        help="Provide the Azure blob URL for json file")
    
    parser.add_argument("--blob-name", type=str,
                        help="Provide the Azure blob name for json file")
    
    parser.add_argument("--blob-size", type=int,
                        help="Provide the Azure blob size for json file")
    
    parser.add_argument("--blob-metadata-url", type=str,
                        help="Provide the Azure blob URL for metadata file")
    
    parser.add_argument("--use-call-recording", action="store_true",
                        help="Use call recording given via acs")
    
    parser.add_argument("--transcription-output-container", type=str, help="Name of the container to save the transcription results")

    parser.add_argument("--telephone", action="store_true", help="If the file is from a telephone recording")

    parser.add_argument("--convert_audio", action="store_true", default=False, help="Convert the audio file to mp3 and sample rate 16000 and bitrate 32") 

    parser.add_argument("--load_audio", action="store_true", default=False, help="Load the audio file into memory") 

    parser.add_argument("--task", type=str, default="transcribe", help="Task to perform on the audio file")

    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams to use")

    parser.add_argument("--language", type=str, default="no", help="Language of the audio file")

    parser.add_argument("--chunk-size", type=int, default=28, help="Chunk size to use for processing")

    parser.add_argument("--return-timestamps", action="store_true", help="Return timestamps for each word")

    # parser.add_argument("--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit")

    return parser.parse_args()



def set_logging_level(log_level: str) -> None:
    log_level_upper = log_level.upper()
    if not hasattr(logging, log_level_upper):
        raise ValueError(f"Invalid log level: {log_level}")
    
    level = getattr(logging, log_level_upper)
    logging.getLogger().setLevel(level)
    logger.info(f"Logging level set to {log_level_upper}")


async def run_app_from_args() -> int:
    """Main async entry point for the application."""
    # Parse command line arguments
    args: argparse.Namespace = parse_arguments()
    AsyncConfigManager(args=args)
    
    # set_logging_level(args.log_level)
    
    if args.run_webapp or os.getenv("RUN_WEBAPP") == "True":
        logger.info("Running Flask web application")
        try:
            if os.getenv("FLASK_ENV") == "production":
                from flask_setup import run_flask_app
                run_flask_app()
            else:
                from flask_setup import run_flask_app_dev
                run_flask_app_dev()
            return 0
        except Exception as e:
            logger.error(f"Failed to run Flask web application: {str(e)}")
            return 1
    
    
    # Check if we're processing multiple files
    elif args.audio_files and len(args.files) > 1:
        try:
            logger.info(f"Processing {len(args.files)} files concurrently")
            results = await process_multiple_files(args.files, args.model)
            logger.info(f"Successfully processed {len(results)} out of {len(args.files)} files")
            return 0
        except Exception as e:
            logger.error(f"Failed to process multiple files: {str(e)}")
            return 1
    
    elif args.audio_file:
        logger.info(f"Processing single audio file: {args.audio_file}")
        # Process a single file
        try:
            if args.convert_audio:
                # Process and convert audio
                if args.bitrate:
                    audio_buffer = await AudioProcessor.convert_audio(args.audio_file, args.bitrate)
                elif args.sample_rate:
                    audio_buffer = await AudioProcessor.convert_audio(args.audio_file, target_sample_rate=args.sample_rate)
                elif args.bitrate and args.sample_rate:
                    audio_buffer = await AudioProcessor.convert_audio(args.audio_file, args.bitrate, args.sample_rate)
                else:
                    audio_buffer = await AudioProcessor.convert_audio(args.audio_file)
            
            if args.load_audio:
                audio_data, sample_rate = await AudioProcessor.load_audio_for_model(audio_buffer)

            transcriber = AsyncWhisperTranscriber(
                model_path=args.model, 
                args=args, 
                audio_data=audio_data or args.audio_file, 
                sample_rate=sample_rate)
            result = await transcriber.transcribe()
            logger.info(f"Successfully processed file: {args.audio_file}")
            return 0
        except Exception as e:
            logger.error(f"Failed to read file {args.audio_file}: {str(e)}")
            return 1
        
    elif args.json_file:
        logger.info(f"Processing single json file: {args.audio_file}")
        # Process a single file
        try:
            with open(args.json_file, "r") as f:
                global global_json_data
                global_json_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read file {args.json_file}: {str(e)}")
            return 1
    elif args.use_call_recording:
        logger.info(f"Processing call recording")
        # Process a single file
        transcriber = AsyncWhisperTranscriber(
                model_path=args.model,
                transcription_container=args.transcription_output_container
                )
        result = await transcriber.transcribe()
        # Save results
        result_filename = await transcriber.save_results(result)
        blob_storage_service = BlobStorageService(config=AsyncConfigManager())
        await blob_storage_service.upload_to_transcriptions_blob_storage(result_file_path=result_filename)
        logger.info(f"Successfully processed file")
        return 0
    else:
        logger.error("Please specify either --use-call-recording or --use-webapp")
        return 1
    

        

def import_modules():
    logger.info("Importing required modules...")
    try:
        # Try to import required packages
        import importlib
        for package in ["aiofiles", "huggingface_hub", "transformers", "aiohttp"]:
            try:
                importlib.import_module(package)
            except ImportError:
                logger.warning(f"{package} not found, attempting to install...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Successfully installed {package}")
    except Exception as e:
        logger.warning(f"Could not auto-install dependencies: {str(e)}")

def run_app(data=None):
    logger.info("Running app...")
    return asyncio.run(run_app_from_args())


def main():
    """Main entry point for the application."""
    logger.info("Starting application...")
    try:
        return run_app()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130  # Standard Unix exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())