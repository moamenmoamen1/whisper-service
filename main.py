"""
سيرفر بسيط يستقبل رابط فيديو، يفرّغ الصوت لنص باستخدام faster-whisper (مجاني ومفتوح المصدر)،
ويرجّع النص الكامل + توقيت كل جملة (segments).
مصمم للنشر على Railway/Render (طبقة مجانية).
"""

import tempfile
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper transcription service")

# نموذج "small" اختيار متوازن بين السرعة والدقة، ومناسب لموارد الطبقة المجانية
# لو حابب دقة أعلى وعندك موارد أكتر، غيّرها لـ "medium" أو "large-v3"
model = WhisperModel("small", device="cpu", compute_type="int8")


class TranscribeRequest(BaseModel):
    video_url: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "whisper-transcription"}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_file:
            urllib.request.urlretrieve(req.video_url, tmp_file.name)

            segments_generator, info = model.transcribe(tmp_file.name, beam_size=5)

            segments = []
            full_text_parts = []
            for seg in segments_generator:
                segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })
                full_text_parts.append(seg.text.strip())

        return {
            "transcript": " ".join(full_text_parts),
            "segments": segments,
            "language": info.language,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"فشل التفريغ الصوتي: {str(e)}")
