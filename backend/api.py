from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging
from pathlib import Path
import librosa

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 180
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Emotion_detection", "src")))

from .test_model import predict_emotion

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://music-emotion-recognition-6354-oxwuj6zf4.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file must be 25 MB or smaller.")

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, Path(file.filename or "upload").name)

    try:
        bytes_written = 0
        with open(temp_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Audio file must be 25 MB or smaller.",
                    )
                buffer.write(chunk)

        logger.info("Audio uploaded: %s bytes; extracting duration", bytes_written)
        duration = librosa.get_duration(path=temp_path)
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=413,
                detail="Audio must be 3 minutes or shorter.",
            )

        logger.info("Starting emotion prediction for %.1f-second audio", duration)
        return predict_emotion(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
