import os
import argparse
from typing import Optional
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient as AsyncSecretClient


# User defined imports
from ContainerName import ContainerName
from logger_config import get_logger, setup_logging
import logging
setup_logging(logging.INFO)
logger = get_logger(__name__)


class AsyncConfigManager:
    """Handles configuration and credentials for the application with async support."""
    _instance = None  # Class variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        """Override __new__ to ensure only one instance of AsyncConfigManager exists."""
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self,
                 web_app: bool = False,
                 audio_file: bool = False,
                 audio_files: bool = False,
                 telephone_json_data: bool = False,
                 json_data_from_telephone: bool = False,
                 args: Optional[argparse.Namespace] = None
                 ):
        """Initialize the configuration manager."""
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        logger.info("Initializing AsyncConfigManager")
        self.web_app = web_app or os.getenv("WEB_APP", False)
        self.container_env = os.getenv("CONTAINER_ENV")
        self.AZURE_STORAGE_BLOB_ACCOUNT_ENDPOINT = os.getenv("AZURE_STORAGE_BLOB_ACCOUNT_URL")
        self.AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.RECORDINGS_CALL_DATA_CONTAINER_NAME = os.getenv("RECORDINGS_CALL_DATA_CONTAINER_NAME") or ContainerName.RECORDINGS_CALL_DATA.value
        self.RECORDINGS_CONTAINER_NAME = os.getenv("RECORDINGS_CONTAINER_NAME") or ContainerName.RECORDINGS.value
        self.TRANSCRIPTIONS_CONTAINER_NAME = args.transcription_output_container or os.getenv("TRANSCRIPTIONS_CONTAINER_NAME") or ContainerName.TRANSCRIPTS.value
        self.audio_file = audio_file
        self.audio_files = audio_files
        self.telephone_json_data = telephone_json_data
        self.json_data_from_telephone = json_data_from_telephone
        self.blob_uri = os.getenv("BLOB_URI") or args.blob_uri
        self.blob_name = os.getenv("BLOB_NAME") or args.blob_name
        self.azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
        self.storage_secret_name = os.getenv("STORAGE_SECRET_NAME")
        self.acs_secret_name = os.getenv("ACS_SECRET_NAME")
        self.cloud_env = os.getenv("CLOUD_ENV")
        self.args = args
        self.default_credential = AsyncDefaultAzureCredential()
        # Set up AsyncSecretClient for Azure Key Vault
        self.secret_client = AsyncSecretClient(vault_url=self.key_vault_url, credential=self.default_credential)
        # self.acs_connection_string_task = asyncio.create_task(self.get_acs_connection_string())

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = AsyncConfigManager()
        return cls._instance
    
    async def get_acs_connection_string(self):
        """Retrieve the ACS connection string from Azure Key Vault."""
        logger.info("Retrieving ACS Connection String...")
        try:
            # Await the async get_secret method
            secret = await self.secret_client.get_secret(self.acs_secret_name)
            self.acs_connection_string = secret.value
            logger.info("ACS Connection String retrieved successfully.")
        except Exception as e:
            logger.error(f"Error retrieving ACS secret: {e}")
            self.acs_connection_string = os.getenv("ACS_CONNECTION_STRING")

        return self.acs_connection_string

