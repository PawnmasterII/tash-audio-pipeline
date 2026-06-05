"""Download + unpack the small English Vosk model into ./models.

No key, no signup. Run once:  python download_vosk_model.py
"""
import os
import urllib.request
import zipfile

import config

URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def main() -> None:
    models_dir = os.path.dirname(config.VOSK_MODEL_PATH)
    os.makedirs(models_dir, exist_ok=True)
    if os.path.isdir(config.VOSK_MODEL_PATH):
        print(f"model already present at {config.VOSK_MODEL_PATH}")
        return

    zip_path = config.VOSK_MODEL_PATH + ".zip"
    print(f"downloading {URL} (~40 MB) ...")
    urllib.request.urlretrieve(URL, zip_path)
    print("unpacking ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(models_dir)
    os.remove(zip_path)
    print(f"done -> {config.VOSK_MODEL_PATH}")


if __name__ == "__main__":
    main()
