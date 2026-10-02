"""FastAPI server exposing the OpenPronounce web UI and JSON API.

Run with: uvicorn server:app --host 0.0.0.0 --port 8000

Deployment settings (all optional, environment variables):

- ``OPENPRONOUNCE_API_KEY`` (or legacy ``API_KEY``): when set, the POST endpoints require an ``X-API-Key`` header with this value.
- ``OPENPRONOUNCE_LANGUAGES``: comma-separated language codes served (default ``en``).
- ``OPENPRONOUNCE_PRELOAD``: load the models at startup (default ``1``); ``/ready`` answers 503 until done.
- ``OPENPRONOUNCE_MAX_CONCURRENCY``: simultaneous analyses (default ``2``); extra requests wait, then get 503.
- ``OPENPRONOUNCE_MAX_UPLOAD_MB`` / ``OPENPRONOUNCE_MAX_AUDIO_SECONDS`` / ``OPENPRONOUNCE_MAX_TEXT_CHARS``: input limits.
- ``OPENPRONOUNCE_THREADS``: torch intra-op threads (default: torch's own default).
"""

import hmac
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from openpronounce import __version__, audio, phones, speech
from openpronounce.languages import DEFAULT_LANGUAGE, LANGUAGES, get_language

logger = logging.getLogger("openpronounce.server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API_KEY = os.environ.get("OPENPRONOUNCE_API_KEY") or os.environ.get("API_KEY", "")
ENABLED_LANGUAGES = [c.strip() for c in os.environ.get("OPENPRONOUNCE_LANGUAGES", "en").split(",") if c.strip()]
PRELOAD = os.environ.get("OPENPRONOUNCE_PRELOAD", "1") not in ("0", "false", "no", "off")
MAX_CONCURRENCY = max(1, int(os.environ.get("OPENPRONOUNCE_MAX_CONCURRENCY", "2")))
MAX_UPLOAD_BYTES = int(float(os.environ.get("OPENPRONOUNCE_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
MAX_AUDIO_SECONDS = float(os.environ.get("OPENPRONOUNCE_MAX_AUDIO_SECONDS", "60"))
MAX_TEXT_CHARS = int(os.environ.get("OPENPRONOUNCE_MAX_TEXT_CHARS", "1000"))
QUEUE_TIMEOUT_SECONDS = 60
TTS_CACHE_MAX_FILES = 500

_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)
_ready = threading.Event()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _configure_threads():
    threads = os.environ.get("OPENPRONOUNCE_THREADS")
    if threads:
        import torch
        torch.set_num_threads(int(threads))


def _preload_models():
    """Load the English word model and the phone model so the first request is not slow."""
    try:
        for code in ENABLED_LANGUAGES:
            speech._load_models(get_language(code).asr_model)
        if phones.is_enabled():
            phones._load_model()
        _ready.set()
        logger.info("Models loaded, ready")
    except Exception:
        logger.exception("model preload failed")
        raise


@asynccontextmanager
async def lifespan(app):
    if not API_KEY:
        logger.warning("OPENPRONOUNCE_API_KEY is not set: the API is open to everyone")
    _configure_threads()
    if PRELOAD:
        # In a thread so the server answers /health while the models load.
        threading.Thread(target=_preload_models, daemon=True).start()
    else:
        _ready.set()
    yield


app = FastAPI(
    title="OpenPronounce",
    description="Phoneme-level pronunciation assessment (Wav2Vec2 + DTW). English by default, see /languages.",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def require_api_key(key: str = Depends(_api_key_header)):
    """Reject the request unless it carries the configured ``X-API-Key`` (no-op when no API key is configured)."""
    if API_KEY and not (key and hmac.compare_digest(key.encode(), API_KEY.encode())):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


class _Slot:
    """Context manager limiting the number of simultaneous analyses."""

    def __enter__(self):
        if not _slots.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
            raise HTTPException(status_code=503, detail="Server is busy, try again in a moment")

    def __exit__(self, *exc):
        _slots.release()


def _save_upload(upload: UploadFile) -> str:
    """Write the upload to a temp file, enforcing the size limit. Returns the path."""
    suffix = os.path.splitext(upload.filename or "")[1][:10] or ".webm"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="openpronounce-upload-")
    size = 0
    try:
        with os.fdopen(fd, "wb") as buffer:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
                buffer.write(chunk)
    except BaseException:
        _remove(path)
        raise
    return path


def _load_upload(upload: UploadFile):
    """Decode the upload to a 16 kHz mono waveform, enforcing the duration limit. Temp files are removed."""
    path = _save_upload(upload)
    try:
        sound = audio.load(path)
    finally:
        _remove(path)
    if len(sound) / audio.TARGET_SR > MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=413, detail=f"Recording too long (max {MAX_AUDIO_SECONDS:g} s)")
    return sound


def _validate_lang(lang: str) -> str:
    try:
        code = get_language(lang).code
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if code not in ENABLED_LANGUAGES:
        raise HTTPException(status_code=422, detail=f"Language '{code}' is not enabled on this server")
    return code


def _validate_text(text: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=422, detail=f"Text too long (max {MAX_TEXT_CHARS} characters)")
    return text


def _prune_tts_cache():
    """Keep the TTS cache below ``TTS_CACHE_MAX_FILES`` files by deleting the oldest ones."""
    try:
        files = [os.path.join(audio.CACHE_DIR, f) for f in os.listdir(audio.CACHE_DIR) if f.startswith("tts-")]
        files.sort(key=os.path.getmtime)
        for path in files[:max(0, len(files) - TTS_CACHE_MAX_FILES)]:
            _remove(path)
    except OSError:
        pass


# The endpoints below are plain ``def`` on purpose: FastAPI runs them in a thread pool,
# so a long inference does not block the event loop (and the health check).

@app.post("/pronunciation", dependencies=[Depends(require_api_key)])
def api_analyze_pronunciation(file: UploadFile = File(...), expected_text: str = Form(...),
                              lang: str = Form(DEFAULT_LANGUAGE)):
    """Score ``file`` against ``expected_text`` in ``lang``. Returns the full analysis (score, errors, prosody)."""
    lang = _validate_lang(lang)
    _validate_text(expected_text)
    try:
        sound = _load_upload(file)
        with _Slot():
            return speech.compare_audio_with_text(sound, expected_text, lang=lang)
    except HTTPException:
        raise
    except Exception:
        logger.exception("pronunciation analysis failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/speech2text", dependencies=[Depends(require_api_key)])
def api_speech2text(file: UploadFile = File(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Transcribe ``file`` with the Wav2Vec2 model of ``lang``."""
    lang = _validate_lang(lang)
    try:
        sound = _load_upload(file)
        with _Slot():
            return {"transcript": speech.transcribe(sound, lang)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/phonemes", dependencies=[Depends(require_api_key)])
def api_phonemes(text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Return the IPA phonemes of ``text`` in ``lang`` and the word each phoneme belongs to."""
    lang = _validate_lang(lang)
    _validate_text(text)
    try:
        phonemes, words = speech.get_phonemes_with_word_mapping(text, lang)
        return {"phonemes": phonemes, "words": list(words.values())}
    except Exception:
        logger.exception("phonemization failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/tts", dependencies=[Depends(require_api_key)])
def api_tts(background: BackgroundTasks, text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Return a 16 kHz wav reference pronunciation of ``text`` in ``lang``."""
    lang = _validate_lang(lang)
    _validate_text(text)
    try:
        path = audio.text2speech(text, lang=lang)
    except Exception:
        logger.exception("tts failed")
        raise HTTPException(status_code=500, detail="Something went wrong")
    background.add_task(_prune_tts_cache)
    return FileResponse(path, media_type="audio/wav")


@app.get("/languages")
async def api_languages():
    """List the languages enabled on this server (``code`` is the value of the ``lang`` form field)."""
    languages = [{"code": language.code, "name": language.name}
                 for language in LANGUAGES.values() if language.code in ENABLED_LANGUAGES]
    default = DEFAULT_LANGUAGE if DEFAULT_LANGUAGE in ENABLED_LANGUAGES else languages[0]["code"]
    return {"default": default, "languages": languages}


@app.get("/health")
async def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: 200 once the models are loaded, 503 before."""
    if not _ready.is_set():
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ready"}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})
