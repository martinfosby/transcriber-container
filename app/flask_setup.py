import asyncio
import importlib.util
from flask import Flask, request, jsonify
import platform
import importlib
import sys

# User defined imports
from AsyncConfigManager import AsyncConfigManager
from AsyncWhisperTranscriber import AsyncWhisperTranscriber
from BlobStorageService import BlobStorageService

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    if request.method == 'GET':
        return jsonify({'status': 'success'}), 200

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        data = request.get_json()
        app.logger.info(f"Received webhook data: {data}")

        # Do something with the data (e.g., save to DB, trigger a job, etc.)
        return jsonify({'status': 'success'}), 200
    
@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe():
    if request.method == 'POST':
        try:
            data = request.get_json()
            app.logger.info(f"Received recording data: {data}")
            AsyncConfigManager().telephone_json_data = data  # Create an singleton instance of AsyncConfigManager
            AsyncConfigManager().json_data_from_telephone = data.get("json_data_from_telephone", False)
        except Exception as e:
            app.logger.error(f"Error processing request: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
        
        try:
            app.logger.info("Transcribing...")
            model = AsyncConfigManager().args.model
            async def async_transcribe():
                transcriber = AsyncWhisperTranscriber(model_path=model)
                result = await transcriber.transcribe()
                # Save results
                result_filename = await transcriber.save_results(result)
                blob_storage_service = BlobStorageService(config=AsyncConfigManager())
                await blob_storage_service.upload_to_transcriptions_blob_storage(result_file_path=result_filename)
                return result
            transcription = asyncio.run(asyncio.wait_for(async_transcribe(),timeout=AsyncConfigManager().args.timeout)) # run transcription for max 10 minutes
        except Exception as e:
            app.logger.error(f"Error transcribing: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
        
        return jsonify({'status': 'success', "transcription": transcription}), 200

    elif request.method == 'GET':
        return jsonify({'status': 'success', "transcription": ""}), 200

def run_flask_app_dev():
    args = AsyncConfigManager().args
    app.logger.info(f"Host: {args.host}")
    app.logger.info(f"Port: {args.port}")
    app.logger.info(f"Debug: {args.debug}")
    app.run(host=args.host, port=args.port, debug=args.debug)

def run_flask_app():
    args = AsyncConfigManager().args

    system = platform.system()

    if system == "Windows":
        try:
            package = "waitress"
            waitress = importlib.import_module(package)
            app.logger.info("Running Flask app with Waitress on Windows")
            waitress.serve(app, host=args.host, port=args.port, debug=AsyncConfigManager().args.debug)
        except ModuleNotFoundError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            app.logger.info(f"Successfully installed {package}")
            waitress = importlib.import_module(package)
            app.logger.info("Running Flask app with Waitress on Windows")
            waitress.serve(app, host=args.host, port=args.port, debug=AsyncConfigManager().args.debug)
    elif system == "posix":
        try:
            # Check if Gunicorn is available
            gunicorn = importlib.util.find_spec("gunicorn")
            if gunicorn is not None:
                app.logger.info("Gunicorn found. Running with Gunicorn...")
                # Run Gunicorn via subprocess (in production)
                subprocess.run(["gunicorn", "yourapp:app", "-b", "0.0.0.0:8000"])
            else:
                app.logger.warning("Gunicorn not installed. Install Gunicorn using: pip install gunicorn")
                # Fallback to Flask's dev server (only in demo/development)
                app.run(host=args.host, port=args.port, debug=args.debug)
        except Exception as e:
            app.logger.error(f"Error importing Gunicorn: {e}")
            # In case of error with Gunicorn, fallback to Flask dev server (only for local dev)
            app.run(host=args.host, port=args.port, debug=args.debug)
    else :
        app.logger.info("Running Flask app using built-in development server")
        app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    app.run()
