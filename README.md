# Telegram Gemini Voice Bot — Polling + Whisper ASR

ASR: **faster-whisper** (локальная транскрипция), пост-обработка текста: **Gemini** (саммари + ТЗ).  
Никаких вебхуков — чистый **long polling**.

## Шаги деплоя (Render)
1. Залей архив как Web Service (Python).
2. Build: `pip install -r requirements.txt`
3. Start: `python bot.py`
4. Переменные окружения:
   - `TELEGRAM_BOT_TOKEN`
   - `GEMINI_API_KEY`
   - `GROUP_CHAT_ID`
   - `MEETING_URL` (опционально)
   - `WHISPER_MODEL_NAME` (по умолчанию `small`), `WHISPER_DEVICE` (`cpu`/`cuda`), `WHISPER_COMPUTE` (`int8`)

> Примечание: модели `medium/large-v2` требуют больше памяти/CPU и будут медленнее на free-плане.

## Локально (Docker)
```bash
docker build -t tg-gemini-bot-whisper .
docker run -e TELEGRAM_BOT_TOKEN=... -e GEMINI_API_KEY=... -e GROUP_CHAT_ID=-100... -p 8080:8080 tg-gemini-bot-whisper
```

## Логика
- Telegram voice/audio → ffmpeg (OGG → WAV 16k mono) → faster-whisper → текст
- Текст → Gemini → саммари и черновик ТЗ
- Итог → пост в группу + ответ автору
