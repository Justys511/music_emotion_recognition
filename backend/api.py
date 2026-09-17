from fastapi import BackgroundTasks, FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging
import threading
import uuid
from pathlib import Path
import librosa

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 180
logger = logging.getLogger(__name__)
jobs = {}
jobs_lock = threading.Lock()

app = FastAPI()

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    print(f"REQUEST START {request.method} {request.url.path}", flush=True)
    try:
        response = await call_next(request)
        print(
            f"REQUEST END {request.method} {request.url.path} {response.status_code}",
            flush=True,
        )
        return response
    except Exception:
        logger.exception("REQUEST FAILED %s %s", request.method, request.url.path)
        raise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Emotion_detection", "src")))

from .test_model import predict_emotion

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

def process_prediction(job_id, temp_path):
    try:
        print(f"Starting background prediction {job_id}", flush=True)
        result = predict_emotion(temp_path)
        with jobs_lock:
            if result.get("error"):
                jobs[job_id] = {"status": "failed", "error": result["error"]}
            else:
                jobs[job_id] = {"status": "completed", "result": result}
    except Exception as error:
        logger.exception("Prediction failed for job %s", job_id)
        with jobs_lock:
            jobs[job_id] = {"status": "failed", "error": str(error)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict")
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
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

        print(
            f"Audio uploaded: {bytes_written} bytes; extracting duration",
            flush=True,
        )
        duration = librosa.get_duration(path=temp_path)
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=413,
                detail="Audio must be 3 minutes or shorter.",
            )

        job_id = uuid.uuid4().hex
        with jobs_lock:
            jobs[job_id] = {"status": "processing"}
        background_tasks.add_task(process_prediction, job_id, temp_path)
        return {"job_id": job_id, "status": "processing"}
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

@app.get("/predict/{job_id}")
def prediction_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Prediction job not found.")
    return job
