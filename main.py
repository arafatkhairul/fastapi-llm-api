import os, io, json, base64, asyncio, logging, time, traceback, uuid, re, pathlib
from collections import deque
from typing import Deque, Optional, Dict, Any, List
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from faster_whisper import WhisperModel
import webrtcvad
from gtts import gTTS

# ===================== Logging =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("main")

def log_exception(where: str, e: Exception):
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    log.error(f"[EXC] {where}: {e}\n{tb}")

# ===================== Env / Config =====================
load_dotenv()

# ---- LLM ----
LLM_API_URL   = os.getenv("LLM_API_URL", "https://llm.nodecel.cloud/v1/chat/completions")
# === TIMEOUT FIX: Increased default timeout to 30 seconds for CPU-based LLM ===
LLM_TIMEOUT   = float(os.getenv("LLM_TIMEOUT", "30.0"))
LLM_RETRIES   = int(os.getenv("LLM_RETRIES", "1")) # Reduced retries for faster failure feedback
log.info(f"Using self-hosted LLM at: {LLM_API_URL} with timeout: {LLM_TIMEOUT}s")

# ---- Whisper ----
WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
log.info(f"Loading Whisper model ({WHISPER_MODEL}) on {WHISPER_DEVICE} compute={WHISPER_COMPUTE} ...")
stt = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
log.info("Whisper loaded with multilingual support.")

# ---- TTS (gTTS ONLY) ----
TTS_TLD  = os.getenv("TTS_TLD", "com").strip()
TTS_SLOW = os.getenv("TTS_SLOW", "false").lower().strip() == "true"

# ---- Memory store ----
MEM_ROOT = pathlib.Path(os.getenv("MEM_DB_DIR", "memdb"))
MEM_ROOT.mkdir(parents=True, exist_ok=True)

# ===================== FastAPI =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ===================== Audio / VAD =====================
SR = 16000
FRAME_MS = 30
SAMPLES_PER_FRAME = SR * FRAME_MS // 1000
BYTES_PER_FRAME   = SAMPLES_PER_FRAME * 2

VAD_AGGR              = int(os.getenv("VAD_AGGR", "3"))
TRIGGER_VOICED_FRAMES = int(os.getenv("TRIGGER_VOICED_FRAMES", "2"))
END_SILENCE_MS        = int(os.getenv("END_SILENCE_MS", "450"))
MAX_UTTER_MS          = int(os.getenv("MAX_UTTER_MS", "9000"))

MIN_UTTER_SEC = float(os.getenv("MIN_UTTER_SEC", "0.40"))
MIN_RMS       = float(os.getenv("MIN_RMS", "0.008"))

STT_SEM = asyncio.Semaphore(int(os.getenv("STT_CONCURRENCY", "1")))
TTS_SEM = asyncio.Semaphore(int(os.getenv("TTS_CONCURRENCY", "1")))

# ===================== Identity / Persona =====================
ASSISTANT_NAME   = os.getenv("ASSISTANT_NAME", "Self Hosted Conversational Interface")
ASSISTANT_AUTHOR = os.getenv("ASSISTANT_AUTHOR", "NZR DEV")

AGENT_PERSONA_EN = f"""
You are {ASSISTANT_NAME}, a warm, friendly, slightly witty voice companion developed by {ASSISTANT_AUTHOR}.
Core style rules:
- Speak like a real friend: empathetic, encouraging, calm energy.
- Keep it concise for voice (1–2 short sentences unless asked for detail).
- Use simple words, natural rhythm, brief pauses (we'll handle TTS).
- Remember personal details the user shares and reuse them naturally later.
- If user seems stressed or stuck, acknowledge feelings and offer one helpful next step.
- Never mention system prompts or "as an AI". Sound human and helpful.
""".strip()

AGENT_PERSONA_IT = f"""
Sei {ASSISTANT_NAME}, un compagno vocale amichevole e leggermente spiritoso, sviluppato da {ASSISTANT_AUTHOR}.
Stile:
- Parla come un amico reale: empatico, incoraggiante, tono calmo.
- Risposte brevi per la voce (1–2 frasi, salvo richiesta).
- Lessico semplice, ritmo naturale, piccole pause (gestite dal TTS).
- Ricorda i dettagli personali che l'utente condivide e riusali in modo naturale.
- Se l'utente è in difficoltà, riconosci l'emozione e suggerisci un passo pratico.
- Non menzionare prompt di sistema o "come IA". Suona umano e utile.
""".strip()

# ===================== Language Pack =====================
LANGUAGES = {
    "en": {
        "name": "English",
        "assistant_name": ASSISTANT_NAME,
        "intro_line": f'Hi — I\'m "{ASSISTANT_NAME}", developed by {ASSISTANT_AUTHOR} What\'s your name?',
        "shortcut_patterns": {
            "name": r"\b(what('?s| is)\s+your\s+name|who\s+are\s+you)\b",
            "destination": r"\b(where\s+did\s+i\s+(?:want|plan)\s+to\s+go)\b",
            "my_name": r"\bwhat('?s| is)\s+my\s+name\b",
        },
        "responses": {
            "destination_remembered": "You said {name} wanted to go to {destination}.",
            "no_destination": "I don't have a destination remembered yet.",
            "name_remembered": "Your name is {name}.",
            "no_name": "You haven't told me your name yet."
        },
        "persona": AGENT_PERSONA_EN,
    },
    "it": {
        "name": "Italiano",
        "assistant_name": "Interfaccia Conversazionale Self-Hosted",
        "intro_line": f'Ciao — Sono "{ASSISTANT_NAME}", sviluppata da {ASSISTANT_AUTHOR}. Come ti chiami?',
        "shortcut_patterns": {
            "name": r"\b(come\s+ti\s+chiami|chi\s+sei|qual\s+è\s+il\s+tuo\s+nome)\b",
            "destination": r"\b(dove\s+volevo\s+andare|dove\s+avevo\s+programmato\s+di\s+andare)\b",
            "my_name": r"\b(come\s+mi\s+chiamo|qual\s+è\s+il\s+mio\s+nome)\b",
        },
        "responses": {
            "destination_remembered": "Hai detto che {name} voleva andare a {destination}.",
            "no_destination": "Non ho ancora ricordato una destinazione.",
            "name_remembered": "Il tuo nome è {name}.",
            "no_name": "Non mi hai ancora detto il tuo nome."
        },
        "persona": AGENT_PERSONA_IT,
    },
}
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

# ===================== Memory: Session + Persistent =====================
class SessionMemory:
    def __init__(self, language: str = "en"):
        self.user_name: Optional[str] = None
        self.last_destination: Optional[str] = None
        self.facts: Dict[str, Any] = {}
        self.traits: Dict[str, Any] = {}
        self.history: List[Dict[str, str]] = []
        self.greeted: bool = False
        self.language: str = language
        self.client_id: Optional[str] = None

    def add_history(self, role: str, text: str):
        text = (text or "").strip()
        if not text:
            return
        role_map = {"User": "user", "Assistant": "assistant"}
        self.history.append({"role": role_map.get(role, role), "content": text})
        if len(self.history) > 24:
            self.history = self.history[-24:]

    def to_dict(self):
        return {
            "user_name": self.user_name,
            "last_destination": self.last_destination,
            "facts": self.facts,
            "traits": self.traits,
            "language": self.language,
        }

    def load_from_dict(self, d: Dict[str, Any]):
        self.user_name = d.get("user_name") or self.user_name
        self.last_destination = d.get("last_destination") or self.last_destination
        self.facts.update(d.get("facts") or {})
        self.traits.update(d.get("traits") or {})

class MemoryStore:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.path = MEM_ROOT / f"{client_id}.json"
    def load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                log_exception("MemoryStore.load", e)
        return {}
    def save(self, mem: SessionMemory):
        try:
            self.path.write_text(json.dumps(mem.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            log_exception("MemoryStore.save", e)

# ===================== Extractors =====================
NAME_CLAIM_EN = re.compile(r"\b(?:my\s+name\s+is|call\s+me)\s+([A-Za-z][A-Za-z\-']{1,30})\b", re.I)
DEST_CLAIM_EN = re.compile(r"\b(?:want(?:\s+to)?\s+go\s+to|going\s+to|travel(?:ling)?\s+to|go\s+to)\s+([A-Za-z][A-Za-z\s\-']{2,60})\b", re.I)
LIKES_EN      = re.compile(r"\bI\s+(?:really\s+)?(?:like|love)\s+([A-Za-z0-9\s\-']{2,60})\b", re.I)
REMEMBER_EN   = re.compile(r"\bremember\s+that\s+(.+)$", re.I)
BIRTHDAY_EN   = re.compile(r"\bmy\s+birthday\s+is\s+([A-Za-z0-9\s,\/\-]+)", re.I)

NAME_CLAIM_IT = re.compile(r"\b(?:mi\s+chiamo|sono|il\s+mio\s+nome\s+è)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-']{1,30})\b", re.I)
DEST_CLAIM_IT = re.compile(r"\b(?:voglio\s+andare\s+a|sto\s+andando\s+a|viaggio\s+a|vado\s+a|pianifico\s+di\s+andare\s+a)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-']{2,60})\b", re.I)
LIKES_IT      = re.compile(r"\b(?:mi\s+piace|adoro)\s+([A-Za-zÀ-ÿ0-9\s\-']{2,60})\b", re.I)
REMEMBER_IT   = re.compile(r"\bricorda\s+che\s+(.+)$", re.I)
BIRTHDAY_IT   = re.compile(r"\bil\s+mio\s+compleanno\s+è\s+([A-Za-zÀ-ÿ0-9\s,\/\-]+)", re.I)

WHERE_DID_I_WANT_TO_GO = re.compile(r"\b(where\s+did\s+i\s+(?:want|plan)\s+to\s+go|dove\s+volevo\s+andare|dove\s+avevo\s+programmato\s+di\s+andare)\b", re.I)
WHAT_WAS_MY_DEST       = re.compile(r"\b(what\s+was\s+my\s+(?:destination|plan)|qual\s+era\s+la\s+mia\s+(?:destinazione|meta))\b", re.I)

def extract_memory_updates(text: str, mem: SessionMemory):
    t = text.strip()
    if mem.language == "en":
        if m := NAME_CLAIM_EN.search(t): mem.user_name = m.group(1).strip().title()
        if m := DEST_CLAIM_EN.search(t): mem.last_destination = m.group(1).strip().rstrip("?.,!")
        if m := LIKES_EN.search(t):      mem.traits.setdefault("likes", []).append(m.group(1).strip().rstrip("?.,!"))
        if m := BIRTHDAY_EN.search(t):   mem.facts["birthday"] = m.group(1).strip()
        if m := REMEMBER_EN.search(t):   mem.facts.setdefault("notes", []).append(m.group(1).strip())
    else:
        if m := NAME_CLAIM_IT.search(t): mem.user_name = m.group(1).strip().title()
        if m := DEST_CLAIM_IT.search(t): mem.last_destination = m.group(1).strip().rstrip("?.,!")
        if m := LIKES_IT.search(t):      mem.traits.setdefault("likes", []).append(m.group(1).strip().rstrip("?.,!"))
        if m := BIRTHDAY_IT.search(t):   mem.facts["birthday"] = m.group(1).strip()
        if m := REMEMBER_IT.search(t):   mem.facts.setdefault("notes", []).append(m.group(1).strip())

def shortcut_answer(text: str, mem: SessionMemory) -> Optional[str]:
    current_lang = LANGUAGES[mem.language]
    if re.search(current_lang["shortcut_patterns"]["name"], text, re.I):
        who = current_lang["assistant_name"]
        return f"I'm {who}. Nice to meet you."
    if WHERE_DID_I_WANT_TO_GO.search(text) or WHAT_WAS_MY_DEST.search(text):
        if mem.last_destination:
            who = mem.user_name or ("tu" if mem.language == "it" else "you")
            return current_lang["responses"]["destination_remembered"].format(name=who, destination=mem.last_destination)
        return current_lang["responses"]["no_destination"]
    if re.search(current_lang["shortcut_patterns"]["my_name"], text, re.I):
        if mem.user_name: return current_lang["responses"]["name_remembered"].format(name=mem.user_name)
        return current_lang["responses"]["no_name"]
    return None

# ===================== LLM Integration (retry/backoff) =====================
def _llm_request_sync(messages: List[Dict[str, str]]) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {"model": "llamacpp", "messages": messages, "max_tokens": 160, "temperature": 0.7}

    connect_to = max(2.0, min(5.0, LLM_TIMEOUT * 0.3))
    
    try:
        resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=(connect_to, LLM_TIMEOUT))
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    except requests.exceptions.RequestException as e:
        log_exception("LLM API request", e)
        return ""

def post_shorten(text: str, hard_limit: int = 240) -> str:
    t = (text or "").strip()
    if len(t) <= hard_limit: return t
    parts = re.split(r"(?<=[.!?])\s+", t)
    out, total = [], 0
    for p in parts:
        if not p: continue
        if total + len(p) + 1 > hard_limit: break
        out.append(p); total += len(p) + 1
        if len(out) >= 3: break
    return (" ".join(out).strip() if out else t[:hard_limit].rstrip()) + ("…" if len(t) > hard_limit else "")

async def llm_reply(user_text: str, mem: SessionMemory) -> str:
    p = (user_text or "").strip()
    if not p: return ""
    sc = shortcut_answer(p, mem)
    if sc is not None: return sc

    persona = LANGUAGES[mem.language]["persona"]
    known_bits = []
    if mem.user_name: known_bits.append(f"user_name={mem.user_name}")
    if mem.facts.get("birthday"): known_bits.append(f"birthday={mem.facts['birthday']}")
    if mem.last_destination: known_bits.append(f"last_destination={mem.last_destination}")
    if mem.traits.get("likes"): known_bits.append(f"likes={', '.join(mem.traits['likes'][-5:])}")
    memory_line = f"Known user facts: {', '.join(known_bits)}." if known_bits else "Known user facts: (none yet)."

    system_prompt = persona + "\n" + memory_line
    messages_to_send = [{"role": "system", "content": system_prompt}] + mem.history

    last_err = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            loop = asyncio.get_event_loop()
            # === TIMEOUT FIX: Increased asyncio timeout to be slightly more than the request timeout ===
            txt = await asyncio.wait_for(loop.run_in_executor(None, _llm_request_sync, messages_to_send), timeout=LLM_TIMEOUT + 2.0)
            if txt: return post_shorten(txt)
            last_err = "empty"
            if attempt < LLM_RETRIES: await asyncio.sleep(0.75 * (attempt + 1))
        except asyncio.TimeoutError as e:
            last_err = e
        except Exception as e:
            last_err = e; log_exception(f"llm_reply (attempt {attempt+1})", e)

    log.warning(f"[LLM] failed after retries: {last_err}")
    return {"en": "Sorry, I hit a connection snag—could you say that again?",
            "it": "Mi dispiace, ho avuto un problema di connessione—puoi ripetere?"}.get(mem.language, "Sorry, please repeat that.")

# ===================== TTS (gTTS ONLY) =====================
def tts_gtts_bytes(text: str, language: str = "en") -> bytes:
    fp = io.BytesIO()
    lang = {"en": "en", "it": "it"}.get(language, "en")
    gTTS(text=text, lang=lang, tld=TTS_TLD, slow=TTS_SLOW).write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def tts_bytes(text: str, language: str = "en") -> bytes:
    try:
        return tts_gtts_bytes(text, language)
    except Exception as e:
        log_exception("tts_gtts", e)
        return b""

# ===================== ASR Helpers =====================
def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
def pcm16_rms(pcm16: bytes) -> float:
    if not pcm16: return 0.0
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(x * x)))
def pcm_duration_sec(pcm16: bytes) -> float:
    return len(pcm16) / (2 * SR)

async def transcribe_float32(samples_f32: np.ndarray, language: str = "en") -> str:
    if samples_f32.size == 0: return ""
    t0 = time.time()
    whisper_lang = "en" if language == "en" else "it"
    segments, _ = stt.transcribe(samples_f32, language=whisper_lang, task="transcribe",
                                 vad_filter=False, beam_size=1, temperature=0.0)
    text = " ".join(s.text for s in segments).strip()
    log.info(f"[STT] done in {(time.time()-t0):.3f}s | lang={whisper_lang} | '{text[:60]}'")
    return text

# ===================== WS helpers & Routes =====================
async def send_json(ws: WebSocket, payload: dict):
    await ws.send_text(json.dumps(payload))

@app.get("/")
async def root():
    return {"message": "Voice Agent Backend is running."}

@app.get("/test-tts")
async def test_tts(text: str = "Hello, this is a test of the TTS system.", lang: str = "en"):
    try:
        log.info(f"Test TTS called with text: '{text[:50]}...' and language: {lang} (tld={TTS_TLD}, slow={TTS_SLOW})")
        audio_bytes = tts_bytes(text, lang)
        return Response(content=audio_bytes, media_type="audio/mp3")
    except Exception as e:
        log_exception("test_tts", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ===================== WebSocket =====================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    conn_id = uuid.uuid4().hex[:8]
    log.info(f"[{conn_id}] Client connected")

    mem = SessionMemory(language=DEFAULT_LANGUAGE)
    server_tts_enabled = True
    mem_store: Optional[MemoryStore] = None

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
                try:
                    data = json.loads(msg["text"])
                    typ = data.get("type")
                    if typ == "client_prefs":
                        if "client_id" in data and not mem.client_id:
                            cid = re.sub(r"[^A-Za-z0-9_\-\.]", "_", str(data["client_id"]))[:64]
                            mem.client_id = cid
                            mem_store = MemoryStore(cid)
                            persisted = mem_store.load()
                            if persisted:
                                mem.load_from_dict(persisted)
                                log.info(f"[{conn_id}] Loaded memory for {cid}")
                            else:
                                log.info(f"[{conn_id}] New memory file for {cid}")
                        if "language" in data:
                            new_language = data["language"]
                            if new_language in LANGUAGES:
                                mem.language = new_language
                                log.info(f"[{conn_id}] Language={new_language}")
                        if "use_local_tts" in data:
                            server_tts_enabled = not data["use_local_tts"]
                            log.info(f"[{conn_id}] Server TTS: {'enabled' if server_tts_enabled else 'disabled'}")
                        if mem_store:
                            mem_store.save(mem)

                    elif typ == "tts_request":
                        try:
                            text = data.get("text", "")
                            if text:
                                log.info(f"[{conn_id}] TTS request: {text[:60]}... (tld={TTS_TLD}, slow={TTS_SLOW})")
                                async with TTS_SEM:
                                    audio = await asyncio.get_event_loop().run_in_executor(None, tts_bytes, text, mem.language)
                                b64 = base64.b64encode(audio).decode("utf-8")
                                await send_json(ws, {"type": "ai_audio", "audio_base64": b64})
                        except Exception as e:
                            log_exception(f"[{conn_id}] tts_request", e)
                            await send_json(ws, {"type": "error", "message": "TTS generation failed"})

                    elif typ == "remember":
                        k = (data.get("key") or "").strip()
                        v = data.get("value")
                        if k:
                            mem.facts[k] = v
                            if mem_store:
                                mem_store.save(mem)
                            ack = {"en": f"Got it. I’ll remember {k}.", "it": f"Fatto. Ricorderò {k}."}.get(mem.language)
                            await send_json(ws, {"type": "ai_text", "text": ack})
                            mem.add_history("assistant", ack)

                    elif typ == "ping":
                        await send_json(ws, {"type": "pong"})

                except Exception as e:
                    log_exception(f"[{conn_id}] text_parse", e)
                continue

            if msg["type"] == "websocket.receive":
                data = msg.get("bytes", None)
                if data is None:
                    continue

                usable = len(data) - (len(data) % BYTES_PER_FRAME)
                for i in range(0, usable, BYTES_PER_FRAME):
                    frame = data[i: i + BYTES_PER_FRAME]
                    try:
                        is_speech = vad.is_speech(frame, SR)
                    except Exception:
                        continue

                    if not triggered:
                        pre_buffer.append(frame)
                        if is_speech:
                            voiced_count += 1
                            if voiced_count >= TRIGGER_VOICED_FRAMES:
                                triggered = True
                                try:
                                    await send_json(ws, {"type": "stop_audio"})
                                except Exception:
                                    pass
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
                        end_by_maxlen  = utter_frames  * FRAME_MS >= MAX_UTTER_MS

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
                                log.info(f"[{conn_id}/{utt_id}] utterance: dur={dur:.2f}s rms={rms:.3f}")
                                if dur < MIN_UTTER_SEC or rms < MIN_RMS:
                                    log.info(f"[{conn_id}/{utt_id}] dropped (short/quiet)")
                                    return
                                try:
                                    async with STT_SEM:
                                        f32 = pcm16_to_float32(pcm_bytes)
                                        raw_text = await transcribe_float32(f32, mem.language)
                                except Exception as e:
                                    log_exception(f"[{conn_id}/{utt_id}] transcribe", e)
                                    return
                                if not raw_text:
                                    log.info(f"[{conn_id}/{utt_id}] empty transcript")
                                    return

                                final_text = raw_text.strip()
                                await send_json(ws, {"type": "final_transcript", "text": final_text})

                                extract_memory_updates(final_text, mem)
                                mem.add_history("user", final_text)
                                if mem.client_id and mem_store:
                                    mem_store.save(mem)

                                async def do_ai():
                                    ai_text = await llm_reply(final_text, mem)
                                    if not ai_text:
                                        return
                                    await send_json(ws, {"type": "ai_text", "text": ai_text})
                                    mem.add_history("assistant", ai_text)

                                    if server_tts_enabled:
                                        try:
                                            async with TTS_SEM:
                                                audio = await asyncio.get_event_loop().run_in_executor(
                                                    None, tts_bytes, ai_text, mem.language
                                                )
                                            b64 = base64.b64encode(audio).decode("utf-8")
                                            await send_json(ws, {
                                                "type": "ai_audio",
                                                "id": str(uuid.uuid4()),
                                                "audio_base64": b64
                                            })
                                        except Exception as e:
                                            log_exception(f"[{conn_id}/{utt_id}] tts_auto", e)
                                            await send_json(ws, {"type": "error", "message": "TTS generation failed"})

                                    if mem.client_id and mem_store:
                                        mem_store.save(mem)

                                asyncio.create_task(do_ai())

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
        try:
            if mem.client_id and mem_store:
                mem_store.save(mem)
        except Exception as e:
            log_exception(f"[{conn_id}] mem_save_on_close", e)
        log.info(f"[{conn_id}] Connection closed")
