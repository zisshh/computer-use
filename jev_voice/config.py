from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Jev is reachable two ways with an identical request and response shape: TypeSafe's own
# API, or OpenRouter's decisions endpoint (Jev is a "decisions" model there, not a chat
# model -- it rejects /chat/completions). JEV_PROVIDER picks which; the key you have decides.
TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

_PROVIDERS = {
    #             decisions endpoint                            warm-up URL                             default model
    "typesafe":   ("https://api.typesafe.ai/v1/systemone",      "https://api.typesafe.ai/v1/models",    "jev-latest"),
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "https://openrouter.ai/api/v1/credits", "~typesafe/jev-latest"),
}

JEV_PROVIDER = os.environ.get("JEV_PROVIDER", "typesafe" if TYPESAFE_API_KEY else "openrouter").lower()
if JEV_PROVIDER not in _PROVIDERS:
    raise SystemExit(f"Unknown JEV_PROVIDER={JEV_PROVIDER!r} (use {' or '.join(_PROVIDERS)})")

_url, _warmup, _model = _PROVIDERS[JEV_PROVIDER]
JEV_URL = os.environ.get("JEV_URL", _url)
JEV_WARMUP_URL = _warmup
JEV_MODEL = os.environ.get("JEV_MODEL", _model)
JEV_API_KEY = TYPESAFE_API_KEY if JEV_PROVIDER == "typesafe" else OPENROUTER_API_KEY
JEV_KEY_VAR = "TYPESAFE_API_KEY" if JEV_PROVIDER == "typesafe" else "OPENROUTER_API_KEY"

# This key must only ever buy Jev. OpenRouter has no per-key model allowlist, and the
# decisions endpoint already refuses chat models ("does not exist"), but a mistyped
# JEV_URL or JEV_MODEL could still point the key at a billable chat model. Fail loudly.
if JEV_PROVIDER == "openrouter":
    if "/decisions" not in JEV_URL:
        raise SystemExit(
            f"JEV_URL={JEV_URL!r} is not OpenRouter's decisions endpoint. "
            "Jev is a decisions model; chat endpoints would bill a different model."
        )
    if "jev" not in JEV_MODEL.lower():
        raise SystemExit(
            f"JEV_MODEL={JEV_MODEL!r} is not a Jev model. Use ~typesafe/jev-latest."
        )

WHISPER_MODEL = Path(os.environ.get("WHISPER_MODEL", ROOT / "models" / "ggml-base.en.bin"))
WHISPER_PORT = int(os.environ.get("WHISPER_PORT", "8178"))
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "6"))
# "en" forces English. A multilingual model also accepts "auto", or a code like
# "hi". Hinglish is code-switched -- English structure with Hindi words in it --
# so which of these transcribes "khat" correctly is a measured question, not an
# obvious one. An .en model ignores this entirely; it cannot hear Hindi at all.
WHISPER_LANG = os.environ.get("WHISPER_LANG", "en")

# Speech to text: "whisper" (local whisper.cpp, ~100 ms, nothing leaves the machine)
# or "wispr" (Wispr Flow's REST API -- more accurate on names, but a network round trip
# and your audio leaves the machine).
STT_BACKEND = os.environ.get("STT_BACKEND", "whisper").lower()
WISPR_API_KEY = os.environ.get("WISPR_API_KEY", "")
WISPR_URL = os.environ.get("WISPR_URL", "https://platform-api.wisprflow.ai/api/v1/dash/api")
# Feed installed app names to the recogniser so "Spotify" stops coming back as "spot if I am".
STT_BIAS_VOCAB = os.environ.get("STT_BIAS_VOCAB", "1") not in ("0", "false", "no")

SAMPLE_RATE = 16000
# Microphone, by name fragment ("USB Condenser"). Empty = the system default.
# The built-in mic sits on top of the built-in speakers, so it hears your music
# about 7 dB louder than a desk mic does -- measured, not guessed.
MIC = os.environ.get("MIC", "")

# Which browser a website opens in. This machine's default http handler is Velja,
# a URL router -- hand it a URL and the page lands wherever Velja's rules say.
BROWSER = os.environ.get("BROWSER", "Arc")
TTS_VOICE = os.environ.get("TTS_VOICE", "Samantha")
TTS_RATE = int(os.environ.get("TTS_RATE", "210"))

# Confidence gates (tune on your own usage; see docs.typesafe.ai/confidence)
ACTION_MIN_CONFIDENCE = float(os.environ.get("ACTION_MIN_CONFIDENCE", "0.35"))
YES = 0.6
