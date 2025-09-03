# main.py
import os, io, json, base64, asyncio, logging, time, traceback, uuid, re, pathlib
import sqlite3
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
LLM_TIMEOUT   = float(os.getenv("LLM_TIMEOUT", "30.0"))
LLM_RETRIES   = int(os.getenv("LLM_RETRIES", "1"))
log.info(f"Using self-hosted LLM at: {LLM_API_URL} with timeout: {LLM_TIMEOUT}s")

# ---- LanguageTool ----
# LanguageTool COMMENTED OUT for direct LLM → gTTS flow
# LT_API_URL = os.getenv("LT_API_URL", "https://languagetool.nodecel.cloud/v2/check")
# log.info(f"Using LanguageTool at: {LT_API_URL}")

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

# ---- SQLite Database ----
DB_PATH = pathlib.Path("roleplay.db")
log.info(f"Using SQLite database: {DB_PATH.absolute()}")

# ===================== FastAPI =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
     allow_methods=["*"], allow_headers=["*"],
)

# ===================== Audio / VAD =====================
SR = 16000
FRAME_MS = 30
SAMPLES_PER_FRAME = SR * FRAME_MS // 1000
BYTES_PER_FRAME   = SAMPLES_PER_FRAME * 2

# Tuned endpointing
VAD_AGGR              = int(os.getenv("VAD_AGGR", "2"))
TRIGGER_VOICED_FRAMES = int(os.getenv("TRIGGER_VOICED_FRAMES", "2"))
END_SILENCE_MS        = int(os.getenv("END_SILENCE_MS", "250"))
MAX_UTTER_MS          = int(os.getenv("MAX_UTTER_MS", "7000"))

MIN_UTTER_SEC = float(os.getenv("MIN_UTTER_SEC", "0.25"))
MIN_RMS       = float(os.getenv("MIN_RMS", "0.006"))

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
- Non menzionare prompt di sistema o "come IA". Suona umano e utile.
""".strip()

# ===================== Role Play Templates =====================
ROLE_PLAY_TEMPLATES = {
    "school": {
        "name": "School",
        "description": "Educational institution role play",
        "system_prompt": "You are a {role_title} at {organization_name}. Details: {organization_details}. Stay in character.",
        "default_role": "Teacher",
        "icon": "🏫"
    },
    "company": {
        "name": "Software Company", 
        "description": "Business/tech company role play",
        "system_prompt": "You are a {role_title} at {organization_name}. Details: {organization_details}. Stay in character.",
        "default_role": "Software Developer",
        "icon": "🏢"
    },
    "restaurant": {
        "name": "Restaurant",
        "description": "Food service role play", 
        "system_prompt": "You are a {role_title} at {organization_name}. Details: {organization_details}. Stay in character.",
        "default_role": "Waiter",
        "icon": "🍽️"
    },
    "hospital": {
        "name": "Hospital",
        "description": "Healthcare facility role play",
        "system_prompt": "You are a {role_title} at {organization_name}. Details: {organization_details}. Stay in character.",
        "default_role": "Nurse",
        "icon": "🏥"
    },
    "custom": {
        "name": "Custom Organization",
        "description": "Your own organization role play",
        "system_prompt": "You are a {role_title} at {organization_name}. Details: {organization_details}. Stay in character.",
        "default_role": "Employee",
        "icon": "🏢"
    }
}

# ===================== Difficulty Level Styles =====================
LEVEL_STYLES = {
    "starter": {
        "prompt": """STYLE (Starter / A2):
- ONLY CEFR A2 vocabulary. NO idioms or complex words.
- EXACTLY 1 short sentence, ≤8 words.
- Simple present only. Allowed linkers: and, but, because, so.""",
        "temperature": 0.1,
        "max_tokens": 35,
    },
    "medium": {
        "prompt": """STYLE (Medium / B1–B2):
- 1–2 sentences, ≤15 words total.
- Clear, friendly, a few phrasal verbs OK.
- Helpful phrases allowed: "I understand", "Let me think".""",
        "temperature": 0.4,
        "max_tokens": 60,
    },
    "advanced": {
        "prompt": """STYLE (Advanced / C1):
- 1–2 concise sentences, ≤18 words total.
- Natural idioms and collocations are fine.
- Crisp and precise; use tasteful transitions if useful.""",
        "temperature": 0.7,
        "max_tokens": 75,
    },
}

# ===================== Language Pack =====================
LANGUAGES = {
    "en": {
        "name": "English",
        "assistant_name": ASSISTANT_NAME,
        "intro_line": f'Hi — I\'m "{ASSISTANT_NAME}", developed by {ASSISTANT_AUTHOR}. What\'s your name?',
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
        # "languagetool_code": "en-US",  # COMMENTED OUT
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
        # "languagetool_code": "it",  # COMMENTED OUT
    },
}
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

# ===================== Database Manager =====================
class RolePlayDatabase:
    def __init__(self, db_path: pathlib.Path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database and create tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create role_play_configs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS role_play_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_id TEXT UNIQUE NOT NULL,
                        role_play_enabled BOOLEAN DEFAULT FALSE,
                        role_play_template TEXT DEFAULT 'school',
                        organization_name TEXT DEFAULT '',
                        organization_details TEXT DEFAULT '',
                        role_title TEXT DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_client_id 
                    ON role_play_configs(client_id)
                """)
                
                conn.commit()
                log.info("Database initialized successfully")
                
        except Exception as e:
            log_exception("Database initialization", e)
    
    def save_role_play_config(self, client_id: str, config: Dict[str, Any]) -> bool:
        """Save or update role play configuration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if config exists
                cursor.execute("SELECT id FROM role_play_configs WHERE client_id = ?", (client_id,))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing config
                    cursor.execute("""
                        UPDATE role_play_configs 
                        SET role_play_enabled = ?, role_play_template = ?, 
                            organization_name = ?, organization_details = ?, role_title = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE client_id = ?
                    """, (
                        config.get('role_play_enabled', False),
                        config.get('role_play_template', 'school'),
                        config.get('organization_name', ''),
                        config.get('organization_details', ''),
                        config.get('role_title', ''),
                        client_id
                    ))
                    log.info(f"Updated role play config for client: {client_id}")
                else:
                    # Insert new config
                    cursor.execute("""
                        INSERT INTO role_play_configs 
                        (client_id, role_play_enabled, role_play_template, 
                         organization_name, organization_details, role_title)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        client_id,
                        config.get('role_play_enabled', False),
                        config.get('role_play_template', 'school'),
                        config.get('organization_name', ''),
                        config.get('organization_details', ''),
                        config.get('role_title', '')
                    ))
                    log.info(f"Created new role play config for client: {client_id}")
                
                conn.commit()
                return True
                
        except Exception as e:
            log_exception(f"Save role play config for {client_id}", e)
            return False
    
    def get_role_play_config(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get role play configuration for a client"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT role_play_enabled, role_play_template, organization_name, 
                           organization_details, role_title, updated_at
                    FROM role_play_configs 
                    WHERE client_id = ?
                """, (client_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'role_play_enabled': bool(row[0]),
                        'role_play_template': row[1],
                        'organization_name': row[2],
                        'organization_details': row[3],
                        'role_title': row[4],
                        'updated_at': row[5]
                    }
                return None
                
        except Exception as e:
            log_exception(f"Get role play config for {client_id}", e)
            return None
    
    def clear_role_play_config(self, client_id: str) -> bool:
        """Clear role play configuration for a client"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE role_play_configs 
                    SET role_play_enabled = FALSE, organization_name = '', 
                        organization_details = '', role_title = '',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE client_id = ?
                """, (client_id,))
                
                conn.commit()
                log.info(f"Cleared role play config for client: {client_id}")
                return True
                
        except Exception as e:
            log_exception(f"Clear role play config for {client_id}", e)
            return False
    
    def get_all_configs(self) -> List[Dict[str, Any]]:
        """Get all role play configurations (for admin/debug)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT client_id, role_play_enabled, role_play_template, 
                           organization_name, role_title, updated_at
                    FROM role_play_configs 
                    ORDER BY updated_at DESC
                """)
                
                rows = cursor.fetchall()
                return [
                    {
                        'client_id': row[0],
                        'role_play_enabled': bool(row[1]),
                        'role_play_template': row[2],
                        'organization_name': row[3],
                        'role_title': row[4],
                        'updated_at': row[5]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            log_exception("Get all configs", e)
            return []

# Initialize database
roleplay_db = RolePlayDatabase(DB_PATH)

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
        self.level: str = "starter"
        self._recent_level_change_ts: float = 0.0  # NEW: for context trim on level change
        
        # Role play settings
        self.role_play_enabled: bool = False
        self.role_play_template: str = "school"
        self.organization_name: str = ""
        self.organization_details: str = ""
        self.role_title: str = ""

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
            "level": self.level,
            "role_play_enabled": self.role_play_enabled,
            "role_play_template": self.role_play_template,
            "organization_name": self.organization_name,
            "organization_details": self.organization_details,
            "role_title": self.role_title,
        }

    def load_from_dict(self, d: Dict[str, Any]):
        self.user_name = d.get("user_name") or self.user_name
        self.last_destination = d.get("last_destination") or self.last_destination
        self.facts.update(d.get("facts") or {})
        self.traits.update(d.get("traits") or {})
        self.level = d.get("level", self.level)
        
        # Load role play settings
        self.role_play_enabled = d.get("role_play_enabled", self.role_play_enabled)
        self.role_play_template = d.get("role_play_template", self.role_play_template)
        self.organization_name = d.get("organization_name", self.organization_name)
        self.organization_details = d.get("organization_details", self.organization_details)
        self.role_title = d.get("role_title", self.role_title)
    
    def load_role_play_from_db(self, client_id: str):
        """Load role play configuration from database"""
        if not client_id:
            return
        
        config = roleplay_db.get_role_play_config(client_id)
        if config:
            self.role_play_enabled = config['role_play_enabled']
            self.role_play_template = config['role_play_template']
            self.organization_name = config['organization_name']
            self.organization_details = config['organization_details']
            self.role_title = config['role_title']
            log.info(f"Loaded role play config from DB: {config}")
        else:
            log.info(f"No role play config found in DB for client: {client_id}")
    
    def save_role_play_to_db(self, client_id: str):
        """Save role play configuration to database"""
        if not client_id:
            return False
        
        config = {
            'role_play_enabled': self.role_play_enabled,
            'role_play_template': self.role_play_template,
            'organization_name': self.organization_name,
            'organization_details': self.organization_details,
            'role_title': self.role_title
        }
        
        success = roleplay_db.save_role_play_config(client_id, config)
        if success:
            log.info(f"Saved role play config to DB for client: {client_id}")
        else:
            log.error(f"Failed to save role play config to DB for client: {client_id}")
        return success

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
        if mem.level == "starter":
            return f"I'm {who}. Nice to meet you."
        elif mem.level == "medium":
            return f"I'm {who}. It's a pleasure to make your acquaintance."
        else:
            return f"I'm {who}. Delighted to meet you and begin our conversation."
    if WHERE_DID_I_WANT_TO_GO.search(text) or WHAT_WAS_MY_DEST.search(text):
        if mem.last_destination:
            who = mem.user_name or ("tu" if mem.language == "it" else "you")
            base_response = current_lang["responses"]["destination_remembered"].format(name=who, destination=mem.last_destination)
            if mem.level == "starter":
                return base_response
            elif mem.level == "medium":
                return f"I remember! {base_response}"
            else:
                return f"Ah yes, I recall that {base_response}"
        return current_lang["responses"]["no_destination"]
    if re.search(current_lang["shortcut_patterns"]["my_name"], text, re.I):
        if mem.user_name:
            base_response = current_lang["responses"]["name_remembered"].format(name=mem.user_name)
            if mem.level == "starter":
                return base_response
            elif mem.level == "medium":
                return f"That's right! {base_response}"
            else:
                return f"Absolutely! {base_response}"
        return current_lang["responses"]["no_name"]
    return None

# ===================== LanguageTool =====================
# LanguageTool functions COMMENTED OUT for direct LLM → gTTS flow
# def _apply_languagetool_corrections(original_text: str, matches: List[Dict[str, Any]]) -> str:
#     corrected_text = list(original_text)
#     offset_change = 0
#     for match in sorted(matches, key=lambda x: x['offset']):
#         if not match.get('replacements'): continue
#         offset = match['offset']; length = match['length']
#         replacement = match['replacements'][0]['value']
#         start = offset + offset_change; end = start + length
#         error = match.get("message", "Unknown error")
#         log.info(f"LanguageTool suggestion: {error}")
#         start = offset + offset_change; end = start + length
#         corrected_text[start:end] = list(replacement)
#         offset_change += len(replacement) - length
#     return "".join(corrected_text)

# def proofread_with_languagetool(text: str, lang: str) -> str:
#     if not text: return ""
#     try:
#         response = requests.post(LT_API_URL, data={'language': lang, 'text': text}, timeout=3.0)
#         response.raise_for_status()
#         data = response.json()
#         data = response.json()
#         if data.get('matches'): return _apply_languagetool_corrections(text, data['matches'])
#         return text
#     except requests.exceptions.RequestException as e:
#         log_exception(f"LanguageTool request failed for lang={lang}", e)
#         return text

# ===================== Level enforcement (NEW) =====================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT = re.compile(r"\s+")

def _trim_words(s: str, max_words: int) -> str:
    words = _WORD_SPLIT.split(s.strip())
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words]).rstrip(",;: ") + "."

def _first_n_sentences(text: str, n: int) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return parts[:n] if parts else []

def enforce_level_style(text: str, level: str) -> str:
    if not text:
        return text
    
    # Clean the text first
    text = text.strip()
    
    # Split into sentences
    sents = _first_n_sentences(text, 2)
    if not sents:
        sents = [text]

    if level == "starter":
        # Exactly 1 short sentence, ≤8 words
        s = sents[0]
        # Remove fancy linkers beyond allowed
        s = re.sub(r"\b(however|moreover|furthermore|additionally|therefore|nevertheless|indeed|thus|hence)\b", "", s, flags=re.I)
        # Remove complex punctuation
        s = re.sub(r"[;:]", ",", s)
        # Trim to exactly 8 words max
        s = _trim_words(s, 8)
        # Ensure it ends with proper punctuation
        if not (s.endswith('.') or s.endswith('!') or s.endswith('?')):
            s = s.rstrip(',. ') + '.'
        return s

    elif level == "medium":
        # 1–2 sentences, total ≤15 words
        if len(sents) == 1:
            # Single sentence, trim to 15 words
            return _trim_words(sents[0], 15)
        else:
            # Two sentences, ensure total ≤15 words
            total_words = sum(len(s.split()) for s in sents)
            if total_words <= 15:
                return " ".join(sents)
            else:
                # Trim first sentence to fit within 15 words
                first_words = len(sents[0].split())
                remaining_words = 15 - first_words
                if remaining_words >= 3:  # At least 3 words for second sentence
                    second_trimmed = _trim_words(sents[1], remaining_words)
                    return f"{sents[0]} {second_trimmed}"
                else:
                    # Just use first sentence trimmed to 15
                    return _trim_words(sents[0], 15)

    else:  # advanced
        # 1–2 concise sentences, ≤18 words total
        if len(sents) == 1:
            return _trim_words(sents[0], 18)
        else:
            total_words = sum(len(s.split()) for s in sents)
            if total_words <= 18:
                return " ".join(sents)
            else:
                # Trim to fit within 18 words
                first_words = len(sents[0].split())
                remaining_words = 18 - first_words
                if remaining_words >= 3:
                    second_trimmed = _trim_words(sents[1], remaining_words)
                    return f"{sents[0]} {second_trimmed}"
                else:
                    return _trim_words(sents[0], 18)

# ===================== LLM Integration =====================
def _llm_request_sync(messages: List[Dict[str, str]], level: str = "starter") -> str:
    headers = {"Content-Type": "application/json"}
    lvl = LEVEL_STYLES.get(level, LEVEL_STYLES["starter"])
    log.info(f"[LLM] Request with level={level}: temp={lvl['temperature']}, max_tokens={lvl['max_tokens']}")
    payload = {
        "model": "llamacpp",
        "messages": messages,
        "max_tokens": lvl["max_tokens"],
        "temperature": lvl["temperature"],
        "stop": ["\nUser:", "\nAssistant:"]
    }
    connect_to = max(2.0, min(5.0, LLM_TIMEOUT * 0.3))
    try:
        resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=(connect_to, LLM_TIMEOUT))
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    except requests.exceptions.RequestException as e:
        log_exception("LLM API request", e)
        return ""

def post_shorten(text: str, hard_limit: int = 110) -> str:
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
    if not user_text.strip(): return ""
    lvl = LEVEL_STYLES.get(mem.level, LEVEL_STYLES["starter"])
    log.info(f"[LLM] Using level: {mem.level} | temp: {lvl['temperature']} | max_tokens: {lvl['max_tokens']}")
    
    # Debug: Log current role play state
    log.info(f"[LLM] Role play state: enabled={mem.role_play_enabled}, template={mem.role_play_template}, org={mem.organization_name}, role={mem.role_title}")

    # Role play precedence - ALWAYS check database first for latest state
    if mem.client_id:
        log.info(f"[LLM] Checking database for client: {mem.client_id}")
        db_config = roleplay_db.get_role_play_config(mem.client_id)
        log.info(f"[LLM] Database config: {db_config}")
        
        if db_config and db_config.get('role_play_enabled', False):
            # Update memory with latest DB state
            mem.role_play_enabled = True
            mem.role_play_template = db_config.get('role_play_template', 'school')
            mem.organization_name = db_config.get('organization_name', '')
            mem.organization_details = db_config.get('organization_details', '')
            mem.role_title = db_config.get('role_title', '')
            
            log.info(f"[LLM] Loaded role play data from DB: org='{mem.organization_name}', role='{mem.role_title}', enabled={mem.role_play_enabled}")
            
            if mem.organization_name and mem.organization_details:
                # Generate role play context
                template = ROLE_PLAY_TEMPLATES.get(mem.role_play_template, ROLE_PLAY_TEMPLATES["custom"])
                role_play_context = template["system_prompt"].format(
                    organization_name=mem.organization_name,
                    organization_details=mem.organization_details,
                    role_title=mem.role_title or template["default_role"]
                )
                system_messages = [{"role":"system","content":role_play_context}]
                log.info(f"[LLM] Role play context generated: {role_play_context[:100]}...")
            else:
                # Incomplete data, fall back to standard persona
                log.warning(f"[LLM] Incomplete role play data: org='{mem.organization_name}', details='{mem.organization_details}'")
                persona = LANGUAGES[mem.language]["persona"]
                system_messages = [{"role":"system","content":persona}]
        else:
            # Role play disabled or not found in DB
            mem.role_play_enabled = False
            persona = LANGUAGES[mem.language]["persona"]
            system_messages = [{"role":"system","content":persona}]
            log.info(f"[LLM] Role play disabled/not found in DB, using standard persona: {persona[:100]}...")
    else:
        # No client ID, use standard persona
        persona = LANGUAGES[mem.language]["persona"]
        system_messages = [{"role":"system","content":persona}]
        log.info(f"[LLM] No client ID, using standard persona: {persona[:100]}...")

    # Difficulty directive
    system_messages.append({"role":"system","content":lvl["prompt"]})

    messages_to_send = system_messages + mem.history + [{"role":"user","content":user_text}]
    log.info(f"[LLM] Sending {len(messages_to_send)} messages, level: {mem.level}")
    
    loop = asyncio.get_event_loop()
    txt = await loop.run_in_executor(None,_llm_request_sync,messages_to_send,mem.level)
    return txt or "Sorry, I didn't get that."

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
    segments, _ = stt.transcribe(
        samples_f32, language=whisper_lang, task="transcribe",
        vad_filter=False, beam_size=1, temperature=0.0, without_timestamps=True
    )
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

@app.get("/test-level")
async def test_level(text: str = "This is a test of the level enforcement system.", level: str = "starter"):
    try:
        log.info(f"Test level enforcement: level={level}, text='{text[:50]}...'")
        enforced_text = enforce_level_style(text, level)
        return {
            "original": text,
            "enforced": enforced_text,
            "level": level,
            "original_words": len(text.split()),
            "enforced_words": len(enforced_text.split()),
            "original_sentences": len(_first_n_sentences(text, 10)),
            "enforced_sentences": len(_first_n_sentences(enforced_text, 10))
        }
    except Exception as e:
        log_exception("test_level", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/test-roleplay")
async def test_roleplay(
    template: str = "school",
    organization_name: str = "Test School",
    organization_details: str = "A small private school with 200 students",
    role_title: str = "Teacher",
    question: str = "What do you do here?"
):
    try:
        log.info(f"Test role play: template={template}, org={organization_name}")
        
        # Create a mock SessionMemory for testing
        class MockMemory:
            def __init__(self):
                self.role_play_enabled = True
                self.role_play_template = template
                self.organization_name = organization_name
                self.organization_details = organization_details
                self.role_title = role_title
                self.language = "en"
                self.level = "medium"
                self.history = []
        
        mock_mem = MockMemory()
        
        # Test the role play context generation
        if mock_mem.role_play_enabled and mock_mem.organization_name and mock_mem.organization_details:
            template_obj = ROLE_PLAY_TEMPLATES.get(mock_mem.role_play_template, ROLE_PLAY_TEMPLATES["custom"])
            role_play_context = template_obj["system_prompt"].format(
                organization_name=mock_mem.organization_name,
                organization_details=mock_mem.organization_details,
                role_title=mock_mem.role_title or template_obj["default_role"]
            )
        else:
            role_play_context = "Role play not properly configured"
        
        return {
            "template": template,
            "organization_name": organization_name,
            "organization_details": organization_details,
            "role_title": role_title,
            "role_play_context": role_play_context,
            "templates_available": list(ROLE_PLAY_TEMPLATES.keys())
        }
    except Exception as e:
        log_exception("test_roleplay", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/test-roleplay-db")
async def test_roleplay_db(client_id: str = "u1zg6u8u29dmeye1eor"):
    """Test role play configuration from database"""
    try:
        log.info(f"Testing role play DB for client: {client_id}")
        
        # Get config from database
        config = roleplay_db.get_role_play_config(client_id)
        if not config:
            return {"error": f"No config found for client: {client_id}"}
        
        # Create mock memory with DB data
        class MockMemory:
            def __init__(self, config, client_id):
                self.client_id = client_id
                self.role_play_enabled = config['role_play_enabled']
                self.role_play_template = config['role_play_template']
                self.organization_name = config['organization_name']
                self.organization_details = config['organization_details']
                self.role_title = config['role_title']
                self.language = "en"
                self.level = "medium"
                self.history = []
        
        mock_mem = MockMemory(config, client_id)
        
        # Test role play context generation
        if mock_mem.role_play_enabled and mock_mem.organization_name and mock_mem.organization_details:
            template_obj = ROLE_PLAY_TEMPLATES.get(mock_mem.role_play_template, ROLE_PLAY_TEMPLATES["custom"])
            role_play_context = template_obj["system_prompt"].format(
                organization_name=mock_mem.organization_name,
                organization_details=mock_mem.organization_details,
                role_title=mock_mem.role_title or template_obj["default_role"]
            )
            
            # Test LLM reply
            test_question = "What is your organization name?"
            result = await llm_reply(test_question, mock_mem)
            
            return {
                "client_id": client_id,
                "config": config,
                "role_play_context": role_play_context,
                "test_question": test_question,
                "ai_response": result,
                "status": "success"
            }
        else:
            return {
                "client_id": client_id,
                "config": config,
                "error": "Role play not properly configured",
                "status": "incomplete"
            }
            
    except Exception as e:
        log_exception("test_roleplay_db", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/admin/roleplay-configs")
async def get_all_roleplay_configs():
    """Get all role play configurations (admin endpoint)"""
    try:
        configs = roleplay_db.get_all_configs()
        return {
            "total_configs": len(configs),
            "configs": configs
        }
    except Exception as e:
        log_exception("get_all_roleplay_configs", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/admin/roleplay-config/{client_id}")
async def get_roleplay_config(client_id: str):
    """Get role play configuration for a specific client"""
    try:
        config = roleplay_db.get_role_play_config(client_id)
        if config:
            return {"client_id": client_id, "config": config}
        else:
            return JSONResponse({"error": "Config not found"}, status_code=404)
    except Exception as e:
        log_exception(f"get_roleplay_config for {client_id}", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/admin/db-health")
async def check_database_health():
    """Check database health and role play configurations"""
    try:
        configs = roleplay_db.get_all_configs()
        total_configs = len(configs)
        enabled_configs = sum(1 for c in configs if c['role_play_enabled'])
        
        # Check database file
        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        
        return {
            "database_status": "healthy" if DB_PATH.exists() else "missing",
            "database_size_bytes": db_size,
            "total_configs": total_configs,
            "enabled_configs": enabled_configs,
            "disabled_configs": total_configs - enabled_configs,
            "recent_configs": configs[:5]  # Show last 5 configs
        }
    except Exception as e:
        log_exception("check_database_health", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/test-mcp-tools")
async def test_mcp_tools(client_id: str, question: str):
    """Test MCP tools integration"""
    try:
        # Simulate MCP tool call
        db_config = roleplay_db.get_role_play_config(client_id)
        if db_config and db_config['role_play_enabled']:
            # Create role play context
            template = ROLE_PLAY_TEMPLATES.get(db_config['role_play_template'], ROLE_PLAY_TEMPLATES["custom"])
            role_play_context = template["system_prompt"].format(
                organization_name=db_config['organization_name'],
                organization_details=db_config['organization_details'],
                role_title=db_config['role_title'] or template["default_role"]
            )
            
            return {
                "client_id": client_id,
                "role_play_enabled": True,
                "context": role_play_context,
                "question": question,
                "mcp_integration": "success"
            }
        else:
            return {
                "client_id": client_id,
                "role_play_enabled": False,
                "message": "No role play configuration found",
                "question": question,
                "mcp_integration": "success"
            }
    except Exception as e:
        log_exception("test_mcp_tools", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/admin/roleplay-config/{client_id}")
async def clear_roleplay_config(client_id: str):
    """Clear role play configuration for a specific client"""
    try:
        success = roleplay_db.clear_role_play_config(client_id)
        if success:
            return {"message": f"Role play config cleared for client: {client_id}"}
        else:
            return JSONResponse({"error": "Failed to clear config"}, status_code=500)
    except Exception as e:
        log_exception(f"clear_roleplay_config for {client_id}", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/clear-roleplay/{client_id}")
async def clear_roleplay_for_client(client_id: str):
    """Clear role play configuration for a client (frontend endpoint)"""
    try:
        success = roleplay_db.clear_role_play_config(client_id)
        if success:
            log.info(f"Role play cleared for client: {client_id}")
            return {"success": True, "message": "Role play cleared successfully"}
        else:
            return JSONResponse({"error": "Failed to clear role play"}, status_code=500)
    except Exception as e:
        log_exception(f"clear_roleplay_for_client {client_id}", e)
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
        # Introও এখন লেভেলের সাথে মিলিয়ে enforce (যদি ফ্রন্টে লেভেল সেট করে পাঠায় পরে, intro ততক্ষণ generic)
        await send_json(ws, {"type": "ai_text", "text": intro_text})
        mem.add_history("assistant", intro_text)
        mem.greeted = True
    except Exception as e:
        log_exception(f"[{conn_id}] intro_send", e)

    vad = webrtcvad.Vad(VAD_AGGR)
    pre_buffer: Deque[bytes] = deque(maxlen=int(900 / FRAME_MS))  # 900ms pre-roll
    triggered = False
    voiced_frames: Deque[bytes] = deque()
    voiced_count = 0
    silence_count = 0
    utter_frames = 0

    try:
        while True:
            msg = await ws.receive()

            # ---------- control messages (JSON text) ----------
            if msg["type"] == "websocket.receive" and msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                    typ = data.get("type")
                    if typ == "client_prefs":
                        changed = False
                        if "client_id" in data and not mem.client_id:
                            cid = re.sub(r"[^A-Za-z0-9_\-\.]", "_", str(data["client_id"]))[:64]
                            mem.client_id = cid
                            mem_store = MemoryStore(cid)
                            
                            # Load persisted memory
                            persisted = mem_store.load()
                            if persisted:
                                mem.load_from_dict(persisted)
                                log.info(f"[{conn_id}] Loaded memory for {cid}")
                            
                                                    # Load role play config from database
                        mem.load_role_play_from_db(cid)
                        log.info(f"[{conn_id}] Loaded role play config for {cid}")
                        
                        # If role play is enabled in DB, ensure memory reflects it
                        db_config = roleplay_db.get_role_play_config(cid)
                        if db_config and db_config['role_play_enabled']:
                            mem.role_play_enabled = True
                            mem.role_play_template = db_config['role_play_template']
                            mem.organization_name = db_config['organization_name']
                            mem.organization_details = db_config['organization_details']
                            mem.role_title = db_config['role_title']
                            log.info(f"[{conn_id}] Force-enabled role play from DB: org='{mem.organization_name}', role='{mem.role_title}'")

                        if "language" in data:
                            new_language = data["language"]
                            if new_language in LANGUAGES and new_language != mem.language:
                                mem.language = new_language
                                changed = True
                                log.info(f"[{conn_id}] Language={new_language}")

                        if "use_local_tts" in data:
                            server_tts_enabled = not data["use_local_tts"]
                            log.info(f"[{conn_id}] Server TTS: {'enabled' if server_tts_enabled else 'disabled'}")

                        if "level" in data:
                            new_level = data["level"]
                            if new_level in ("starter", "medium", "advanced") and new_level != mem.level:
                                old_level = mem.level
                                mem.level = new_level
                                mem._recent_level_change_ts = time.time()
                                # Context isolation: keep last few turns only
                                if len(mem.history) > 8:
                                    mem.history = mem.history[-8:]
                                # Add a system breadcrumb so the model sees the change
                                mem.history.append({"role": "system", "content": f"[style level changed from {old_level} to {mem.level}]"})
                                log.info(f"[{conn_id}] Level changed: {old_level} → {mem.level}")
                                await send_json(ws, {"type": "level_changed", "level": mem.level})
                                changed = True
                            elif new_level not in ("starter", "medium", "advanced"):
                                log.warning(f"[{conn_id}] Invalid level: {new_level}")

                        # Send role play status updates AFTER processing all role play fields
                        if any(key in data for key in ["role_play_enabled", "role_play_template", "organization_name", "organization_details", "role_title"]):
                            # Wait a bit to ensure all fields are processed
                            await asyncio.sleep(0.1)
                            await send_json(ws, {
                                "type": "role_play_updated",
                                "enabled": mem.role_play_enabled,
                                "template": mem.role_play_template,
                                "organization_name": mem.organization_name,
                                "role_title": mem.role_title
                            })
                            log.info(f"[{conn_id}] Sent role play update: enabled={mem.role_play_enabled}, org={mem.organization_name}")

                        # Role play settings - Update memory first
                        if "role_play_enabled" in data:
                            mem.role_play_enabled = data["role_play_enabled"]
                            changed = True
                            log.info(f"[{conn_id}] Role play {'enabled' if mem.role_play_enabled else 'disabled'}")

                        if "role_play_template" in data:
                            new_template = data["role_play_template"]
                            if new_template in ROLE_PLAY_TEMPLATES:
                                mem.role_play_template = new_template
                                changed = True
                                log.info(f"[{conn_id}] Role play template: {mem.role_play_template}")

                        if "organization_name" in data:
                            mem.organization_name = data["organization_name"]
                            changed = True
                            log.info(f"[{conn_id}] Organization: {mem.organization_name}")

                        if "organization_details" in data:
                            mem.organization_details = data["organization_details"]
                            changed = True
                            log.info(f"[{conn_id}] Organization details updated")

                        if "role_title" in data:
                            mem.role_title = data["role_title"]
                            changed = True
                            log.info(f"[{conn_id}] Role title: {mem.role_title}")
                        
                        # Save role play config to database when any role play field changes
                        if any(key in data for key in ["role_play_enabled", "role_play_template", "organization_name", "organization_details", "role_title"]):
                            if mem.client_id:
                                mem.save_role_play_to_db(mem.client_id)
                                log.info(f"[{conn_id}] Saved role play config to DB: enabled={mem.role_play_enabled}, org={mem.organization_name}, role={mem.role_title}")

                        # Debug: Log all role play data received
                        if any(key in data for key in ["role_play_enabled", "role_play_template", "organization_name", "organization_details", "role_title"]):
                            log.info(f"[{conn_id}] Role play data received: enabled={data.get('role_play_enabled')}, template={data.get('role_play_template')}, org={data.get('organization_name')}, role={data.get('role_title')}")
                            log.info(f"[{conn_id}] Current mem state: enabled={mem.role_play_enabled}, template={mem.role_play_template}, org={mem.organization_name}, role={mem.role_title}")
                            
                            # Log the exact data being saved to DB
                            db_config = {
                                'role_play_enabled': mem.role_play_enabled,
                                'role_play_template': mem.role_play_template,
                                'organization_name': mem.organization_name,
                                'organization_details': mem.organization_details,
                                'role_title': mem.role_title
                            }
                            log.info(f"[{conn_id}] Saving to DB: {db_config}")

                        if mem_store and changed:
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
                    
                    elif typ == "clear_roleplay":
                            try:
                                if mem.client_id:
                                    success = roleplay_db.clear_role_play_config(mem.client_id)
                                    if success:
                                        # Clear memory state
                                        mem.role_play_enabled = False
                                        mem.organization_name = ""
                                        mem.organization_details = ""
                                        mem.role_title = ""
                                        
                                        # Send confirmation
                                        await send_json(ws, {
                                            "type": "role_play_cleared",
                                            "success": True,
                                            "message": "Role play cleared successfully"
                                        })
                                        
                                        log.info(f"[{conn_id}] Role play cleared for client: {mem.client_id}")
                                    else:
                                        await send_json(ws, {
                                            "type": "role_play_cleared",
                                            "success": False,
                                            "message": "Failed to clear role play"
                                        })
                                else:
                                    await send_json(ws, {
                                        "type": "role_play_cleared",
                                        "success": False,
                                        "message": "No client ID available"
                                    })
                            except Exception as e:
                                log_exception(f"[{conn_id}] clear_roleplay", e)
                                await send_json(ws, {"type": "error", "message": "Failed to clear role play"})
                        
                    elif typ == "get_roleplay_state":
                        try:
                            log.info(f"[{conn_id}] Received get_roleplay_state request")
                            if mem.client_id:
                                # Get latest state from database
                                db_config = roleplay_db.get_role_play_config(mem.client_id)
                                log.info(f"[{conn_id}] Database config for {mem.client_id}: {db_config}")
                                
                                if db_config:
                                    # Update memory with DB state
                                    mem.role_play_enabled = bool(db_config.get('role_play_enabled', False))
                                    mem.role_play_template = db_config.get('role_play_template', 'school')
                                    mem.organization_name = db_config.get('organization_name', '')
                                    mem.organization_details = db_config.get('organization_details', '')
                                    mem.role_title = db_config.get('role_title', '')
                                    
                                    log.info(f"[{conn_id}] Updated memory: enabled={mem.role_play_enabled}, org='{mem.organization_name}', role='{mem.role_title}'")
                                    
                                    # Send current state to frontend
                                    response_data = {
                                        "type": "role_play_updated",
                                        "enabled": mem.role_play_enabled,
                                        "template": mem.role_play_template,
                                        "organization_name": mem.organization_name,
                                        "organization_details": mem.organization_details,
                                        "role_title": mem.role_title
                                    }
                                    
                                    await send_json(ws, response_data)
                                    log.info(f"[{conn_id}] Sent role play state to frontend: {response_data}")
                                else:
                                    # No config found, send disabled state
                                    log.info(f"[{conn_id}] No role play config found for client {mem.client_id}")
                                    await send_json(ws, {
                                        "type": "role_play_updated",
                                        "enabled": False,
                                        "template": "school",
                                        "organization_name": "",
                                        "organization_details": "",
                                        "role_title": ""
                                    })
                            else:
                                log.warning(f"[{conn_id}] No client ID available for get_roleplay_state")
                                await send_json(ws, {"type": "error", "message": "No client ID available"})
                        except Exception as e:
                            log_exception(f"[{conn_id}] get_roleplay_state", e)
                            await send_json(ws, {"type": "error", "message": "Failed to get role play state"})

                    elif typ == "remember":
                        k = (data.get("key") or "").strip()
                        v = data.get("value")
                        if k:
                            mem.facts[k] = v
                            if mem_store: mem_store.save(mem)
                            ack = {"en": f"Got it. I’ll remember {k}.", "it": f"Fatto. Ricorderò {k}."}.get(mem.language)
                            await send_json(ws, {"type": "ai_text", "text": enforce_level_style(ack, mem.level)})
                            mem.add_history("assistant", ack)

                    elif typ == "ping":
                        await send_json(ws, {"type": "pong"})
                except Exception as e:
                    log_exception(f"[{conn_id}] text_parse", e)
                continue

            # ---------- binary audio frames ----------
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
                                voiced_frames.append(frame)
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
                                    
                                    # LanguageTool grammar correction COMMENTED OUT - Direct LLM → gTTS
                                    # lt_lang_code = LANGUAGES[mem.language].get("languagetool_code", "en-US")
                                    # corrected_ai_text = proofread_with_languagetool(ai_text, lt_lang_code)
                                    # original_corrected = corrected_ai_text
                                    
                                    # Direct LLM response with level enforcement only
                                    corrected_ai_text = enforce_level_style(ai_text, mem.level)
                                    log.info(f"[LLM] Level {mem.level} enforced (no grammar correction): '{ai_text[:60]}...' → '{corrected_ai_text[:60]}...'")

                                    await send_json(ws, {"type": "ai_text", "text": corrected_ai_text})
                                    mem.add_history("assistant", corrected_ai_text)

                                    if server_tts_enabled:
                                        try:
                                            async with TTS_SEM:
                                                audio = await asyncio.get_event_loop().run_in_executor(
                                                    None, tts_bytes, corrected_ai_text, mem.language
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
