import json
from pathlib import Path
import os
import asyncio
from typing import Any, List
from azure.storage.blob.aio import BlobServiceClient, BlobClient
import aiofiles

from AsyncConfigManager import AsyncConfigManager
from ContainerName import ContainerName
from logger_config import get_logger, setup_logging
import logging
setup_logging(logging.INFO)
logger = get_logger(__name__)


class BlobStorageService():
    def __init__(self, config: AsyncConfigManager):
        logger.info("Initializing BlobStorageService")
        self.config = config
        self.blob_service_client = BlobServiceClient(
            account_url=config.AZURE_STORAGE_BLOB_ACCOUNT_ENDPOINT,
            credential=config.default_credential if config.cloud_env == "azure" else config.AZURE_STORAGE_ACCOUNT_KEY
        )
        self.recording_container_client = self.blob_service_client.get_container_client(config.RECORDINGS_CONTAINER_NAME)
        self.recording_call_data_container_client = self.blob_service_client.get_container_client(config.RECORDINGS_CALL_DATA_CONTAINER_NAME)
        self.transcription_container_client = self.blob_service_client.get_container_client(config.TRANSCRIPTIONS_CONTAINER_NAME)

    async def upload_to_transcriptions_blob_storage(self, result_file_path):
        """Uploads a file to Azure Blob Storage"""
        logger.info(f"Uploading {result_file_path} to Azure Blob Storage")
        try:
            logger.info(f"Uploading {result_file_path} to Azure Blob Storage")
            
            # Extract just the filename without the path
            blob_name = Path(result_file_path).name
            # Log the blob name we're using
            logger.info(f"Using blob name: {blob_name}")
            
            blob_client: BlobClient = self.transcription_container_client.get_blob_client(blob=blob_name)

            file_size = os.path.getsize(result_file_path)
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # Upload the file
            async with aiofiles.open(result_file_path, "rb") as f:
                data = await f.read()
                await blob_client.upload_blob(data, overwrite=True)

            logger.info(f"File uploaded to Azure Blob Storage: {blob_name}")
        except Exception as e:
            logger.error(f"Error uploading to Azure Blob Storage: {str(e)}")
            # Add more detailed error information
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response headers: {e.response.headers}")
                logger.error(f"Response content: {e.response.text}")
            raise

    async def download_blob_from_container(self, container_name: ContainerName, blob: str) -> str:
        """Downloads a blob from a container in Azure Blob Storage."""
        logger.info(f"Downloading blob {blob} from container {container_name}")
        try:
            match container_name:
                case ContainerName.RECORDINGS:
                    container_name, blob_name = blob.split("/", 1)
                    blob_client = self.recording_container_client.get_blob_client(blob=blob_name)
                case ContainerName.RECORDINGS_CALL_DATA:
                    container_name, blob_name = blob.split("/")
                    blob_client = self.recording_call_data_container_client.get_blob_client(blob=blob_name)
                case _:
                    raise ValueError(f"Unknown container: {container_name}")

            downloaded_folder = Path("telephone_recordings")
            download_file_path = downloaded_folder / blob_name
            download_file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(download_file_path, "wb") as download_file:
                data = await blob_client.download_blob()
                await download_file.write(await data.readall())

            logger.info(f"Downloaded blob to {download_file_path}")
            return str(download_file_path)

        except Exception as e:
            logger.error(f"Error downloading blob {blob}: {str(e)}")
            raise


    async def get_call_recording_and_metadata(self, recording_call_data) -> List[List[Any]]:
        """Retrieves the call recording from Azure Communication Services."""
        logger.info("Retrieving call recording from Azure Communication Services...")
        try:
            tasks = []  # List to hold tasks
            for recordingChunk in recording_call_data.get("recordingStorageInfo", {}).get("recordingChunks", []):
                # Then in your function
                content_task = asyncio.create_task(self.download_blob_content(recordingChunk["contentLocation"]))
                metadata_task = asyncio.create_task(self.download_blob_content(recordingChunk["metadataLocation"]))
                tasks.extend([content_task, metadata_task])  # Add tasks to the list
            
            # Await all tasks concurrently
            content, metadata_json_binary = await asyncio.gather(*tasks)
            metadata = json.loads(metadata_json_binary)
            results = {
                "content": content,
                "metadata": metadata
            }

            return results
        except Exception as e:
            logger.error(f"Failed to get call recording: {str(e)}")
            raise
    
    async def download_blob_content(self, blob_url: str) -> bytes:
        """Downloads the content of a blob from Azure Blob Storage."""
        try:
            logger.info(f"Downloading blob content from URL: {blob_url}")
            blob_client = BlobClient.from_blob_url(blob_url=blob_url, credential=self.config.default_credential if self.config.cloud_env == "azure" else self.config.AZURE_STORAGE_ACCOUNT_KEY)
            stream = await blob_client.download_blob()
            return await stream.readall()
        except Exception as e:
            logger.error(f"Failed to download blob content: {str(e)}")
            raise
        