import os, io, json, base64, asyncio, logging, time, traceback, uuid, re
from collections import deque
from typing import Deque, Optional, Dict, Any, List
import numpy as np
import requests # Added for making HTTP requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from faster_whisper import WhisperModel
import webrtcvad

from gtts import gTTS

# -------- Logging setup --------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
log = logging.getLogger("main")


def log_exception(where: str, e: Exception):
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    log.error(f"[EXC] {where}: {e}\n{tb}")


# -------- Env / Models --------
load_dotenv()

# --- Self-Hosted LLM Configuration ---
LLM_API_URL = "https://llm.nodecel.cloud/v1/chat/completions"
log.info(f"Using self-hosted LLM at: {LLM_API_URL}")

log.info("Loading Whisper model (tiny) on CPU for multilingual support...")
stt = WhisperModel("tiny", device="cpu", compute_type="int8")
log.info("Whisper loaded with multilingual support (English, Italian, etc.)")


# -------- FastAPI --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Audio/VAD config --------
SR = 16000
FRAME_MS = 30
SAMPLES_PER_FRAME = SR * FRAME_MS // 1000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2

VAD_AGGR = 3
TRIGGER_VOICED_FRAMES = 5
END_SILENCE_MS = 600
MAX_UTTER_MS = 15000

MIN_UTTER_SEC = 0.40
MIN_RMS = 0.012


# -------- Assistant identity / phrases --------
ASSISTANT_NAME = "Self Hosted Conversational Interface"
ASSISTANT_AUTHOR = "NZR DEV"
NAME_REPLY = f'I’m "{ASSISTANT_NAME}", developed by {ASSISTANT_AUTHOR}.'
INTRO_LINE = (
    f'Hi — I’m "{ASSISTANT_NAME}", developed by {ASSISTANT_AUTHOR}. What’s your name?'
)

# Quick detect “what is your name / who are you”
NAME_PAT = re.compile(r"\b(what('?s| is)\s+your\s+name|who\s+are\s+you)\b", re.I)

# Language configuration
LANGUAGES = {
    "en": {
        "name": "English",
        "assistant_name": "Self Hosted Conversational Interface",
        "name_reply": f'I\'m "Self Hosted Conversational Interface", developed by {ASSISTANT_AUTHOR}.',
        "intro_line": f'Hi — I\'m "Self Hosted Conversational Interface", developed by {ASSISTANT_AUTHOR}. What\'s your name?',
        "name_pattern": r"\b(what('?s| is)\s+your\s+name|who\s+are\s+you)\b",
        "shortcut_patterns": {
            "name": r"\b(what('?s| is)\s+your\s+name|who\s+are\s+you)\b",
            "destination": r"\b(where\s+did\s+i\s+(?:want|plan)\s+to\s+go)\b",
            "my_name": r"\bwhat('?s| is)\s+my\s+name\b"
        },
        "responses": {
            "destination_remembered": "You said {name} wanted to go to {destination}.",
            "no_destination": "I don't have a destination remembered yet.",
            "name_remembered": "Your name is {name}.",
            "no_name": "You haven't told me your name yet."
        }
    },
    "it": {
        "name": "Italiano",
        "assistant_name": "Interfaccia Conversazionale Self-Hosted",
        "name_reply": f'Sono "Interfaccia Conversazionale Self-Hosted", sviluppata da {ASSISTANT_AUTHOR}.',
        "intro_line": f'Ciao — Sono "Interfaccia Conversazionale Self-Hosted", sviluppata da {ASSISTANT_AUTHOR}. Come ti chiami?',
        "name_pattern": r"\b(come\s+ti\s+chiami|chi\s+sei|qual\s+è\s+il\s+tuo\s+nome)\b",
        "shortcut_patterns": {
            "name": r"\b(come\s+ti\s+chiami|chi\s+sei|qual\s+è\s+il\s+tuo\s+nome)\b",
            "destination": r"\b(dove\s+volevo\s+andare|dove\s+avevo\s+programmato\s+di\s+andare)\b",
            "my_name": r"\b(come\s+mi\s+chiamo|qual\s+è\s+il\s+mio\s+nome)\b"
        },
        "responses": {
            "destination_remembered": "Hai detto che {name} voleva andare a {destination}.",
            "no_destination": "Non ho ancora ricordato una destinazione.",
            "name_remembered": "Il tuo nome è {name}.",
            "no_name": "Non mi hai ancora detto il tuo nome."
        }
    }
}

# Default language
DEFAULT_LANGUAGE = "en"

# -------- Helpers --------


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def pcm16_rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def pcm_duration_sec(pcm16: bytes) -> float:
    return len(pcm16) / (2 * SR)


async def transcribe_float32(samples_f32: np.ndarray, language: str = "en") -> str:
    if samples_f32.size == 0:
        return ""
    t0 = time.time()
    
    # Map language codes to Whisper language codes
    whisper_lang = "en" if language == "en" else "it"
    
    segments, _ = stt.transcribe(
        samples_f32,
        language=whisper_lang,
        task="transcribe",
        vad_filter=False,
        beam_size=1,
        temperature=0.0,
    )
    text = " ".join(s.text for s in segments).strip()
    log.info(f"[STT] done in {(time.time()-t0):.3f}s | lang={whisper_lang} | '{text[:60]}'")
    return text


def tts_bytes(text: str, language: str = "en") -> bytes:
    fp = io.BytesIO()
    # Map language codes to gTTS language codes
    lang_map = {
        "en": "en",
        "it": "it"
    }
    gtts_lang = lang_map.get(language, "en")
    gTTS(text=text, lang=gtts_lang).write_to_fp(fp)
    fp.seek(0)
    return fp.read()


# ---------- Mini session memory ----------
class SessionMemory:
    def __init__(self, language: str = "en"):
        self.user_name: Optional[str] = None
        self.last_destination: Optional[str] = None
        self.facts: Dict[str, Any] = {}
        self.history: List[Dict[str, str]] = []
        self.greeted: bool = False
        self.language: str = language

    def add_history(self, role: str, text: str):
        text = (text or "").strip()
        if not text:
            return
        # Use 'user' and 'assistant' for role, matching OpenAI format
        role_map = {"User": "user", "Assistant": "assistant"}
        self.history.append({"role": role_map.get(role, role), "content": text})
        if len(self.history) > 16:
            self.history = self.history[-16:]


# lightweight extractors for both languages
NAME_CLAIM_EN = re.compile(
    r"\b(?:my\s+name\s+is|call\s+me)\s+([A-Za-z][A-Za-z\-']{1,30})\b", re.I
)
DEST_CLAIM_EN = re.compile(
    r"\b(?:want(?:\s+to)?\s+go\s+to|going\s+to|travel(?:ling)?\s+to|go\s+to)\s+([A-Za-z][A-Za-z\s\-']{2,60})\b",
    re.I,
)

# Italian patterns
NAME_CLAIM_IT = re.compile(
    r"\b(?:mi\s+chiamo|sono|il\s+mio\s+nome\s+è)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-']{1,30})\b", re.I
)
DEST_CLAIM_IT = re.compile(
    r"\b(?:voglio\s+andare\s+a|sto\s+andando\s+a|viaggio\s+a|vado\s+a|pianifico\s+di\s+andare\s+a)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-']{2,60})\b",
    re.I,
)

# Combined patterns for both languages
WHERE_DID_I_WANT_TO_GO = re.compile(
    r"\b(where\s+did\s+i\s+(?:want|plan)\s+to\s+go|dove\s+volevo\s+andare|dove\s+avevo\s+programmato\s+di\s+andare)\b", re.I
)
WHAT_WAS_MY_DEST = re.compile(
    r"\b(what\s+was\s+my\s+(?:destination|plan)|qual\s+era\s+la\s+mia\s+(?:destinazione|meta))\b", re.I
)


def extract_memory_updates(text: str, mem: SessionMemory):
    # Extract name based on language
    if mem.language == "en":
        if m := NAME_CLAIM_EN.search(text):
            name = m.group(1).strip().title()
            mem.user_name = name
        if m := DEST_CLAIM_EN.search(text):
            dest = m.group(1).strip().rstrip("?.,!")
            mem.last_destination = dest
    elif mem.language == "it":
        if m := NAME_CLAIM_IT.search(text):
            name = m.group(1).strip().title()
            mem.user_name = name
        if m := DEST_CLAIM_IT.search(text):
            dest = m.group(1).strip().rstrip("?.,!")
            mem.last_destination = dest


def shortcut_answer(text: str, mem: SessionMemory) -> Optional[str]:
    current_lang = LANGUAGES[mem.language]
    
    # Check name questions
    if re.search(current_lang["shortcut_patterns"]["name"], text, re.I):
        return current_lang["name_reply"]
    
    # Check destination questions
    if WHERE_DID_I_WANT_TO_GO.search(text) or WHAT_WAS_MY_DEST.search(text):
        if mem.last_destination:
            who = mem.user_name or ("tu" if mem.language == "it" else "you")
            return current_lang["responses"]["destination_remembered"].format(name=who, destination=mem.last_destination)
        return current_lang["responses"]["no_destination"]
    
    # Check "what's my name" questions
    if re.search(current_lang["shortcut_patterns"]["my_name"], text, re.I):
        if mem.user_name:
            return current_lang["responses"]["name_remembered"].format(name=mem.user_name)
        else:
            return current_lang["responses"]["no_name"]
    
    return None


# ---------- LLM Integration (Self-Hosted) ---------
def _llm_request_sync(messages: List[Dict[str, str]]) -> str:
    """Synchronous function to make the blocking HTTP request to the self-hosted LLM."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llamacpp",  # This can be any string for llama-cpp-python
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.7,
    }
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=8.0)
        response.raise_for_status()
        data = response.json()
        return (data['choices'][0]['message']['content'] or "").strip()
    except requests.exceptions.RequestException as e:
        log_exception("LLM API request", e)
        return "Sorry, I had trouble connecting to my brain."

def post_shorten(text: str, hard_limit: int = 320) -> str:
    t = (text or "").strip()
    if len(t) <= hard_limit:
        return t
    parts = re.split(r"(?<=[.!?])\s+", t)
    out, total = [], 0
    for p in parts:
        if not p: continue
        if total + len(p) + 1 > hard_limit: break
        out.append(p)
        total += len(p) + 1
        if len(out) >= 3: break
    return (" ".join(out).strip() if out else t[:hard_limit].rstrip()) + ("…" if len(t) > hard_limit else "")

async def llm_reply(
    user_text: str, mem: SessionMemory, timeout: float = 10.0, retries: int = 1
) -> Optional[str]:
    p = (user_text or "").strip()
    if not p: return ""
    
    sc = shortcut_answer(p, mem)
    if sc is not None: return sc

    # Language-specific system prompts
    system_prompts = {
        "en": "You are a warm, friendly voice assistant. Keep replies ultra concise and easy to speak. Default: 1–2 short sentences. Focus only on the user's main intent.",
        "it": "Sei un assistente vocale caloroso e amichevole. Mantieni le risposte ultra concise e facili da pronunciare. Predefinito: 1-2 frasi brevi. Concentrati solo sull'intento principale dell'utente."
    }
    
    system_prompt = system_prompts.get(mem.language, system_prompts["en"])
    
    messages_to_send = [{"role": "system", "content": system_prompt}] + mem.history

    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            loop = asyncio.get_event_loop()
            
            txt = await asyncio.wait_for(
                loop.run_in_executor(None, _llm_request_sync, messages_to_send),
                timeout=timeout,
            )
            
            txt = post_shorten(txt)
            if txt:
                log.info(f"[LLM] ok in {(time.time()-t0):.3f}s | len={len(txt)}")
                return txt
            else:
                log.warning(f"[LLM] received empty response (attempt {attempt+1})")

        except asyncio.TimeoutError:
            log.warning(f"[LLM] timeout after {timeout}s (attempt {attempt+1})")
        except Exception as e:
            log_exception(f"llm_reply (attempt {attempt+1})", e)

    return ""


async def send_json(ws: WebSocket, payload: dict):
    await ws.send_text(json.dumps(payload))


@app.get("/")
async def root():
    return {"message": "Voice Agent Backend is running."}

@app.get("/test-tts")
async def test_tts(text: str = "Hello, this is a test of the TTS system.", lang: str = "en"):
    """Test TTS functionality"""
    try:
        log.info(f"Test TTS called with text: '{text[:50]}...' and language: {lang}")
        audio_bytes = tts_bytes(text, lang)
        log.info(f"TTS generated {len(audio_bytes)} bytes of audio")
        return Response(content=audio_bytes, media_type="audio/mp3")
    except Exception as e:
        log_exception("test_tts", e)
        return {"error": str(e)}


# -------- WS endpoint --------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    conn_id = uuid.uuid4().hex[:8]
    log.info(f"[{conn_id}] Client connected")

    mem = SessionMemory(language=DEFAULT_LANGUAGE)
    server_tts_enabled = True

    try:
        intro_text = LANGUAGES[mem.language]["intro_line"]
        await send_json(ws, {"type": "ai_text", "text": intro_text})
        mem.add_history("assistant", intro_text)
        mem.greeted = True
    except Exception as e:
        log_exception(f"[{conn_id}] intro_send", e)

    vad = webrtcvad.Vad(VAD_AGGR)
    pre_buffer: Deque[bytes] = deque(maxlen=int(300 / FRAME_MS))
    triggered = False
    voiced_frames: Deque[bytes] = deque()
    voiced_count = 0
    silence_count = 0
    utter_frames = 0

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.receive" and msg.get("text") is not None:
                # Handle control messages from client
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "client_prefs":
                        if "language" in data:
                            # Update session memory with new language
                            new_language = data["language"]
                            if new_language in LANGUAGES:
                                mem.language = new_language
                                log.info(f"[{conn_id}] Language changed to: {new_language}")
                            else:
                                log.warning(f"[{conn_id}] Invalid language: {new_language}")
                        
                        if "use_local_tts" in data:
                            server_tts_enabled = not data["use_local_tts"]
                            log.info(f"[{conn_id}] Server TTS: {'enabled' if server_tts_enabled else 'disabled'}")
                        
                        if data.get("type") == "tts_request":
                            try:
                                text = data.get("text", "")
                                if text:
                                    log.info(f"[{conn_id}] TTS request received for text: {text[:50]}...")
                                    audio = await asyncio.get_event_loop().run_in_executor(None, tts_bytes, text, mem.language)
                                    b64 = base64.b64encode(audio).decode("utf-8")
                                    await send_json(ws, {"type": "ai_audio", "audio_base64": b64})
                                    log.info(f"[{conn_id}] TTS request fulfilled, audio sent")
                            except Exception as e:
                                log_exception(f"[{conn_id}] tts_request", e)
                                await send_json(ws, {"type": "error", "message": "TTS generation failed"})
                except Exception as e:
                    log_exception(f"[{conn_id}] client_prefs parse", e)
                continue

            if msg["type"] == "websocket.receive":
                data = msg.get("bytes", None)
                if data is None: continue

                usable = len(data) - (len(data) % BYTES_PER_FRAME)
                for i in range(0, usable, BYTES_PER_FRAME):
                    frame = data[i : i + BYTES_PER_FRAME]
                    try:
                        is_speech = vad.is_speech(frame, SR)
                    except Exception: continue

                    if not triggered:
                        pre_buffer.append(frame)
                        if is_speech:
                            voiced_count += 1
                            if voiced_count >= TRIGGER_VOICED_FRAMES:
                                triggered = True
                                voiced_frames.extend(pre_buffer)
                                pre_buffer.clear()
                                utter_frames = len(voiced_frames)
                                voiced_count = 0
                                silence_count = 0
                        else:
                            voiced_count = 0
                    else:
                        voiced_frames.append(frame)
                        utter_frames += 1
                        if is_speech:
                            silence_count = 0
                        else:
                            silence_count += 1

                        end_by_silence = silence_count * FRAME_MS >= END_SILENCE_MS
                        end_by_maxlen = utter_frames * FRAME_MS >= MAX_UTTER_MS
                        if end_by_silence or end_by_maxlen:
                            utter_pcm = b"".join(voiced_frames)
                            voiced_frames.clear()
                            silence_count = 0
                            utter_frames = 0
                            triggered = False

                            async def process_utterance(pcm_bytes: bytes):
                                utt_id = uuid.uuid4().hex[:6]
                                dur = pcm_duration_sec(pcm_bytes)
                                rms = pcm16_rms(pcm_bytes)
                                log.info(f"[{conn_id}/{utt_id}] utterance ready: dur={dur:.2f}s rms={rms:.3f}")
                                if dur < MIN_UTTER_SEC or rms < MIN_RMS:
                                    log.info(f"[{conn_id}/{utt_id}] dropped (too short/quiet)")
                                    return
                                try:
                                    f32 = pcm16_to_float32(pcm_bytes)
                                    text = await transcribe_float32(f32, mem.language)
                                except Exception as e:
                                    log_exception(f"[{conn_id}/{utt_id}] transcribe", e)
                                    return

                                if not text:
                                    log.info(f"[{conn_id}/{utt_id}] empty transcript")
                                    return

                                extract_memory_updates(text, mem)
                                mem.add_history("user", text)

                                await send_json(ws, {"type": "final_transcript", "text": text})

                                ai_text = await llm_reply(text, mem, timeout=10.0)
                                if not ai_text:
                                    log.warning(f"[{conn_id}/{utt_id}] LLM empty/timeout")
                                    return

                                await send_json(ws, {"type": "ai_text", "text": ai_text})
                                mem.add_history("assistant", ai_text)

                                # Always generate TTS for real conversations (not just test requests)
                                try:
                                    t0 = time.time()
                                    log.info(f"[{conn_id}/{utt_id}] Starting TTS for language: {mem.language}")
                                    audio = await asyncio.get_event_loop().run_in_executor(None, tts_bytes, ai_text, mem.language)
                                    log.info(f"[{conn_id}/{utt_id}] TTS generated, audio size: {len(audio)} bytes")
                                    b64 = base64.b64encode(audio).decode("utf-8")
                                    log.info(f"[{conn_id}/{utt_id}] Base64 encoded, length: {len(b64)}")
                                    await send_json(ws, {"type": "ai_audio", "audio_base64": b64})
                                    log.info(f"[{conn_id}/{utt_id}] TTS audio sent in {(time.time()-t0):.3f}s")
                                except Exception as e:
                                    log_exception(f"[{conn_id}/{utt_id}] tts_auto", e)
                                    log.error(f"[{conn_id}/{utt_id}] TTS failed: {e}")
                                    # Send error message to frontend
                                    await send_json(ws, {"type": "error", "message": f"TTS generation failed: {str(e)}"})

                            asyncio.create_task(process_utterance(utter_pcm))

            elif msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

    except WebSocketDisconnect:
        log.info(f"[{conn_id}] Client disconnected")
    except Exception as e:
        log_exception(f"[{conn_id}] WS loop", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        log.info(f"[{conn_id}] Connection closed")
