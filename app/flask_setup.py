import importlib.util
from flask import Flask, request, jsonify
import platform
import importlib
import sys

# User defined imports
from AsyncConfigManager import AsyncConfigManager
from AsyncWhisperTranscriber import AsyncWhisperTranscriber

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        data = request.get_json()
        app.logger("Received webhook data:", data)

        # Do something with the data (e.g., save to DB, trigger a job, etc.)
        return jsonify({'status': 'success'}), 200
    
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if request.method == 'POST':
        data = request.get_json()
        app.logger("Received recording data:", data)
        AsyncConfigManager().telephone_json_data = data  # Create an singleton instance of AsyncConfigManager
        
        transcriber = AsyncWhisperTranscriber()
        transcription = transcriber.transcribe()

        # Do something with the data (e.g., save to DB, trigger a job, etc.)

        return jsonify({'status': 'success', "transcription": transcription}), 200

def run_flask_app_dev():
    args = AsyncConfigManager.get_instance().args
    app.run(host=args.host, port=args.port, debug=args.debug)

def run_flask_app():
    system = platform.system()

    if system == "Windows":
        try:
            package = "waitress"
            waitress = importlib.import_module(package)
            app.logger("Running Flask app with Waitress on Windows")
            waitress.serve(app, host="127.0.0.1", port=5000, debug=AsyncConfigManager().args.debug)
        except ModuleNotFoundError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            app.logger.info(f"Successfully installed {package}")
            waitress = importlib.import_module(package)
            app.logger("Running Flask app with Waitress on Windows")
            waitress.serve(app, host="127.0.0.1", port=5000, debug=AsyncConfigManager().args.debug)
    elif system == "posix":
        try:
            # Check if Gunicorn is available
            gunicorn = importlib.util.find_spec("gunicorn")
            if gunicorn is not None:
                app.logger("Gunicorn found. Running with Gunicorn...")
                # Run Gunicorn via subprocess (in production)
                subprocess.run(["gunicorn", "yourapp:app", "-b", "0.0.0.0:8000"])
            else:
                app.logger("Gunicorn not installed. Install Gunicorn using: pip install gunicorn")
                # Fallback to Flask's dev server (only in demo/development)
                app.run(host="127.0.0.1", port=5000, debug=True)
        except Exception as e:
            app.logger(f"Error importing Gunicorn: {e}")
            # In case of error with Gunicorn, fallback to Flask dev server (only for local dev)
            app.run(host="127.0.0.1", port=5000, debug=True)
    else :
        app.logger("Running Flask app using built-in development server")
        app.run(host="127.0.0.1", port=5000)

if __name__ == '__main__':
    app.run()
