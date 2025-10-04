import os
import re
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv
import ffmpeg  # audio conversion for Telegram voice (ogg/opus) -> wav
from faster_whisper import WhisperModel
from telegram import Update
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai

# =========================
# Config: polling + Whisper ASR
# =========================
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
GROUP_CHAT_ID_STR  = os.getenv("GROUP_CHAT_ID", "")
MEETING_URL        = os.getenv("MEETING_URL", "https://telemost.yandex.ru/j/85575513867434")

# Whisper model config
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "small")  # tiny/base/small/medium/large-v2
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE", "cpu")        # cpu or cuda
WHISPER_COMPUTE    = os.getenv("WHISPER_COMPUTE", "int8")      # int8/int16/float16/float32

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY or not GROUP_CHAT_ID_STR:
    raise RuntimeError("❌ Missing env vars: TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, GROUP_CHAT_ID")

try:
    GROUP_CHAT_ID = int(GROUP_CHAT_ID_STR)
except ValueError:
    raise RuntimeError("❌ GROUP_CHAT_ID must be an integer (e.g., -1001234567890).")

# Gemini only for text post-processing (summary/tech spec)
genai.configure(api_key=GEMINI_API_KEY)
TEXT_MODEL_NAME = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-flash-latest")
TEXT_MODEL = genai.GenerativeModel(TEXT_MODEL_NAME)

# Init Whisper once at startup
log = logging.getLogger("tg-bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log.info("Loading Whisper model %s (device=%s, compute_type=%s)...", WHISPER_MODEL_NAME, WHISPER_DEVICE, WHISPER_COMPUTE)
_whisper = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
log.info("Whisper loaded. Using Gemini text model: %s", TEXT_MODEL_NAME)

TRIGGER_RE = re.compile(
    r"\b(собер(и|ите)\s+встречу|созв(о|а)н|организу(й|йте)\s+встречу|назнач(ь|ьте)\s+встречу|сдела(й|йте)\s+мит)\b",
    re.IGNORECASE | re.UNICODE,
)

@dataclass
class ProcessedMessage:
    transcript: str
    summary: str
    tech_spec: str
    final_text: str
    meeting_detected: bool


def to_wav16k_mono(ogg_bytes: bytes) -> bytes:
    """Convert OGG/Opus to WAV 16kHz mono s16le via ffmpeg."""
    proc = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    stdout, stderr = proc.communicate(input=ogg_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode('utf-8', errors='ignore')}")
    return stdout


async def whisper_transcribe(wav_bytes: bytes) -> str:
    """Run local Whisper transcription with faster-whisper."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        segments, info = _whisper.transcribe(tmp_path, language="ru")
        text = " ".join(seg.text for seg in segments).strip()
        return text
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


async def gemini_summarize_and_techspec(transcript: str) -> Tuple[str, str]:
    prompt = f"""
Текст речи:
---
{transcript}
---

1) Краткое саммари (5–7 пунктов)
2) Черновик ТЗ: цель, функциональные и нефункциональные требования, сроки, критерии, приемка.
Формат:
### Саммари
<пункты>
### ТЗ
<структурированный текст>
"""
    response = TEXT_MODEL.generate_content(prompt, request_options={"timeout": 90})
    text = (response.text or "").strip()

    summary, tech = "", ""
    if "### ТЗ" in text:
        parts = text.split("### ТЗ", 1)
        summary = parts[0].replace("### Саммари", "").strip()
        tech = parts[1].strip()
    else:
        summary = text
    return summary, tech


def detect_meeting_request(text: str) -> bool:
    return bool(TRIGGER_RE.search(text))


def build_group_message(transcript, summary, tech_spec, meeting_detected):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"📝 *Новый разбор голосового сообщения* (`{ts}`)\n"
    if meeting_detected:
        msg += f"\n🔔 Обнаружен запрос на встречу.\n🔗 [Ссылка на видеовстречу]({MEETING_URL})\n"
    if summary:
        msg += f"\n*Краткое саммари:*\n{summary}\n"
    if tech_spec:
        msg += f"\n*Черновик ТЗ:*\n{tech_spec}\n"
    msg += f"\n<details><summary>Полная транскрипция</summary>\n{transcript}\n</details>"
    return msg


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Пришли голосовое — я сделаю расшифровку и саммари.")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not (msg.voice or msg.audio):
        return

    await msg.chat.send_action(ChatAction.TYPING)
    file = await context.bot.get_file(msg.voice.file_id if msg.voice else msg.audio.file_id)
    ogg_bytes = await file.download_as_bytearray()

    # Convert to WAV and transcribe with Whisper
    try:
        wav_bytes = to_wav16k_mono(bytes(ogg_bytes))
    except Exception as e:
        await msg.reply_text(f"Ошибка конвертации аудио: {e}")
        return

    transcript = await whisper_transcribe(wav_bytes)
    if not transcript:
        await msg.reply_text("Не удалось распознать речь.")
        return

    # Post-process with Gemini (summary + tech spec)
    summary, tech_spec = await gemini_summarize_and_techspec(transcript)
    meeting = detect_meeting_request(transcript)
    final_text = build_group_message(transcript, summary, tech_spec, meeting)

    await context.bot.send_message(
        chat_id=GROUP_CHAT_ID,
        text=final_text,
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )
    await msg.reply_text("✅ Готово! Отправлено в группу.")


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Pure polling
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
