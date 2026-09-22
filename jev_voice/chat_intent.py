"""Turn a spoken sentence into a messaging intent, or None when it is not one.

The decision model has 18 fixed actions and none of them means "message a person", so
every messaging sentence got bent into one that did. "Send the text" planned as typing
the words "the text" and pressing enter -- and that reached a real person. "Send."
alone was dropped as noise. "open my chat with ma on whatsapp" did nothing. "open
Rudra" with WhatsApp focused opened a Notion page. So these sentences are read here, in
code, before the model sees them.

Pure on purpose: a sentence, the focused app's name and the caller's reserved names go
in, a frozen ChatIntent or None comes out. No accessibility, no I/O, nothing borrowed
out of chat.py -- which is what lets every sentence that ever went wrong become a row
in the tests. Finding the chat and reading the compose box stay chat.py's job.

Where two readings are possible, the one that cannot put words in front of a real
person wins: a chat opened rather than a message sent, None rather than a guess.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

CHAT_APPS = ("WhatsApp", "Messages")

# Music or a call leaking into the mic comes back as one run-on transcript. No
# messaging command is that long, and the cap also bounds the regex work on junk.
MAX_WORDS = 60
MAX_CHARS = 600


@dataclass(frozen=True)
class ChatIntent:
    op: str             # "open" | "type" | "send" | "message"
    who: str = ""       # the spoken name, cleaned: "the ziiro group" -> "ziiro"
    app: str = ""       # "WhatsApp" | "Messages" | "" (the caller decides)
    text: str = ""      # the message body, the user's own words
    send: bool = False  # press send after typing

    def __post_init__(self) -> None:
        # Enforced here and not only in parse(): "Send the text" typing "the text" is
        # the one mistake this module must never make again, whoever edits it next.
        if self.op == "send" and (self.text or self.who):
            raise ValueError("a send intent carries nothing to type and nobody to find")


@dataclass(frozen=True)
class _Scene:
    """What is true around the sentence."""

    focused: str                    # the focused app when it is a chat app, else ""
    reserved: frozenset[str]
    named: str = ""                 # an app named earlier in the same sentence


# ------------------------------------------------------------------ vocabulary

_WA = r"whats\s?app|what'?s\s+app|watsapp"
# Whisper hears "WhatsApp" as "what's up" often enough to matter, but "what's up" is
# also the most likely thing anyone sends. It only names the app straight after on/in.
_WA_MISHEARD = r"what'?s\s+up|whats\s?up"
_IM = r"text\s+messages|i\s?messages?|messages"
_PREP = r"(?:on|in|using|via|through|over)"
# "on whatsapp", "in the messages app", and the Hindi order "whatsapp pe". No named
# groups, so it can sit inside other patterns more than once.
_SPEC = (rf"\b(?:{_PREP}\s+(?:my\s+|the\s+)?(?:{_WA}|{_WA_MISHEARD}|{_IM})(?:\s+app)?"
         rf"|(?:{_WA}|{_IM})\s+(?:pe|par|pr|mein|se))\b")
_SPEC_RE = re.compile(_SPEC, re.I)
_SPEC_HEAD = re.compile(rf"^{_SPEC}[\s,]*", re.I)
_SPEC_TAIL = re.compile(rf"[\s,]*{_SPEC}[\s.!?]*$", re.I)
_APP_AS_VERB = re.compile(rf"^(?:{_WA}|i\s?message)\b", re.I)       # "whatsapp rudra ..."
# "my whatsapp chat with ma", and the Hindi verb: "rudra ko whatsapp karo ki main aa
# gaya" named WhatsApp and came back with app="".
_APP_BEFORE_NOUN = re.compile(
    rf"\b(?:{_WA}|i\s?message)\s+(?=(?:text|message|msg|chat|conversation|group|dm|karo"
    r"|kar\s*do|kar\s+de|karna)\b)", re.I)

# Messengers this module does not drive. "message ma on telegram saying hi" read without
# this sent "On telegram saying hi" to ma through WhatsApp.
_ELSEWHERE = re.compile(
    rf"\b{_PREP}\s+(?:my\s+|the\s+)?(?:telegram|slack|discord|signal|instagram|insta"
    r"|messenger|facebook|snapchat|teams|linkedin|twitter|gmail|e-?mail|mail|outlook"
    r"|skype|wechat|viber|sms)\b", re.I)
# main.py already reads "open general chat" as a Discord channel when Discord is in
# focus. With one of these focused and no app named, chat words are about that app.
_OTHER_MESSENGERS = frozenset({"discord", "slack", "telegram", "signal", "messenger",
                               "instagram", "microsoft teams"})

# "Message ma, that was a great dinner": there "that" is the first word of the message.
# It only introduces one when a clause follows it ("that i will be late").
_THAT = (r"that(?!\s+(?:is|was|were|will|would|sounds?|looks?|seems?|works?|should|can"
         r"|could|does|did|has|had|one|thing|guy|girl|place|day|time)\b)")
_SAYING = rf"(?:saying|that\s+says|{_THAT}|ki)(?=[\s,:])"
# What separates the name from the words: "ma, how are you", "govind that i will be
# late". The marker word is tried with its comma: whisper writes "Message ma, saying how
# are you", and with the bare comma tried first "Saying how are you" went out.
_MARK = rf"(?:[\s,:]+{_SAYING}[\s,:]*|\s*[,:]\s*)"
_MARK_RE = re.compile(r"[,:]|\b(?:saying|that|ki)\b", re.I)

_T = r"[^\s,.:;!?]+"                                # one spoken token
# "tell the ziiro group i am late" sent "Group i am late" to the group.
_NAME = rf"(?:(?:my|the)\s+)?{_T}(?:\s+group(?:\s+chat)?)?"
_ART = r"(?:(?:a|an|the|one|ek)\s+)?(?:(?:quick|short)\s+)?"
_MSG_NOUN = r"(?:(?:whatsapp\s+)?(?:text\s+message|message|text|msg))"

_OPEN = (r"(?:open\s+up|open|go\s+to|goto|switch\s+to|jump\s+to|take\s+me\s+to"
         r"|bring\s+up|pull\s+up|show\s+me|show)")
_NOUN = (r"(?:group\s+chat|chats?|conversations?|convo|dms?|thread|text\s+messages"
         r"|messages|texts|group)")
_KHOLO = (r"(?:kholo|khol\s*do|khol\s+de|kholna|khol|open\s+kar\s*do|open\s+karo"
          r"|chalu\s+kar\s*do|chalu\s+karo|start\s+karo|dikhao|dikha\s*do)")
_ACT = r"(?:send|text|message|msg|tell|ask|say|type|write|bolo|likho|bhejo)"

# Never a person. The first nine are the spec's; the rest are what turned up when
# sentences that merely START like a command ("message how are you", "show me new
# messages", "text to speech is not working", "show me messages from today") were read
# as naming someone called "how", "new", "speech" or "today".
_NOT_NAMES = frozenset({
    "me", "him", "her", "them", "it", "this", "that", "someone", "everyone",
    "i", "you", "we", "he", "she", "they", "us", "myself", "yourself", "these", "those",
    "somebody", "everybody", "anyone", "anybody", "nobody", "something", "everything",
    "how", "what", "when", "where", "why", "who", "is", "am", "are", "was", "be", "can",
    "do", "does", "not", "no", "yes", "ok", "okay", "hi", "hello", "hey", "here", "there",
    "to", "for", "of", "on", "in", "at", "from", "saying", "say", "new", "unread", "all",
    "recent", "latest", "last", "any", "some", "text", "voice", "app", "email", "mail",
    "open", "send", "type", "tell", "search", "play", "call", "close", "speech", "box",
    "field", "size", "preview", "tone", "sound", "bubble", "font", "color", "colour",
    "delivered", "received", "deleted", "seen", "sent", "failed", "pending", "today",
    "yesterday", "tomorrow", "tonight", "morning", "week", "month",
    # What gets done to a chat: "delete rudra's chat" opened a chat called "delete rudra".
    "delete", "mute", "unmute", "archive", "unarchive", "clear", "read", "check", "pin",
    "unpin", "block", "unblock", "export", "leave", "exit", "update", "reply", "quit",
    "hide", "minimize", "minimise", "restart", "reload", "refresh", "install", "uninstall",
    # "whatsapp band karo" is about the app, not someone called "band karo".
    "kholo", "khol", "karo", "kar", "karna", "dekho", "dikhao", "band", "chalu", "bhejo",
    "bhej", "likho", "bolo", "batao", "hai",
    # "Text Rahul or Papa" is two people, and "Mujhe Rahul ko message karna hai" ("I
    # need to message Rahul") or "Maine Rahul ko bola" ("I told Rahul") is the user
    # talking, with himself in the sentence.
    "or", "ya", "mujhe", "mujhko", "maine", "humein", "hamein", "hume",
})
# "tell my claude code instance to run the tests" belongs to routing.py, the assistants
# take dictation of their own, and "open instagram messages" is another app's inbox.
# Apps in general arrive through `reserved`; these are the ones chat words get said about.
_NOT_PEOPLE = frozenset({"claude", "claud", "cloud", "clawed", "klaus", "codex", "chatgpt",
                         "gpt", "gemini", "siri", "alexa", "jev", "computer", "terminal",
                         "agent", "google", "whatsapp", "watsapp", "whats", "imessage",
                         "imessages", "discord", "slack", "telegram", "signal", "instagram",
                         "insta", "messenger", "facebook", "fb", "snapchat", "teams",
                         "linkedin", "twitter", "gmail", "outlook", "skype", "wechat",
                         "viber", "sms"})
# Parts of a chat app that sit next to chat words: "open archived chats", "open chat
# settings", "open the first chat", "whatsapp web". Refused on every path.
_CHAT_PARTS = frozenset({"archived", "starred", "pinned", "muted", "blocked", "settings",
                         "info", "media", "notifications", "status", "web", "desktop",
                         "first", "second", "third", "top", "bottom", "next", "previous",
                         "other", "current", "list", "history", "backup"})
_NEVER_WHO = _NOT_NAMES | _NOT_PEOPLE | _CHAT_PARTS
_WHO_NOISE = frozenset({"my", "the", "a", "an", "our", "with", "named", "called", "chat",
                        "chats", "group", "groups", "conversation", "conversations",
                        "convo", "dm", "dms", "message", "messages", "thread", "window",
                        "contact", "please", "now", "quickly"})
# Bare "open X" inside a chat app still has the app's own furniture to reach.
_FURNITURE = frozenset({"preferences", "search", "profile", "calls", "communities",
                        "channels", "updates", "contacts", "menu", "window", "tab",
                        "camera", "emoji", "back", "home", "help", "folder", "page",
                        "sleep", "mode", "dark", "light", "fullscreen", "screen"})

# Words that stand for a message without being one. Sending these is the original bug,
# and it came back as "send this message", "send what i typed", "ye wala message".
_PLACEHOLDER = re.compile(
    r"^(?:(?:(?:the|a|an|this|that|my|your|our|ye|yeh|wo|woh)\s+)?(?:wal[ai]\s+)?"
    r"(?:whatsapp\s+)?(?:text\s+messages?|texts?|messages?|msgs?|draft|reply|something)"
    r"|what(?:ever)?(?:\s+(?:i|i\s+ve|i\s+have|is|s))?\s+(?:typed|wrote|written|said|there)"
    r"|(?:that\s+)?i(?:\s+(?:have|ve|just))?\s+(?:typed|wrote|written|drafted)"
    r"|jo\s+(?:likha|type\s+kiya)(?:\s+hai)?)?$")
_POINTERS = frozenset({"it", "this", "that", "ye", "yeh", "yahi", "isko", "ise", "wo",
                       "woh", "vo", "usko"})
_DETERMINERS = frozenset({"a", "an", "the", "my", "your", "our", "his", "her", "this",
                          "that", "these", "those", "some", "ye", "yeh", "wo", "woh",
                          "mera", "meri", "mere", "apna", "apni"})
# Singular; _a_thing() reads the plural. "send pictures to ma" sent "Pictures".
_THINGS = frozenset({"email", "mail", "file", "photo", "picture", "pic", "pdf", "document",
                     "doc", "link", "screenshot", "video", "image", "location", "contact",
                     "invite", "invoice", "payment", "money", "rupees", "rs", "dollars",
                     "bucks", "request", "audio", "attachment", "resume", "cv", "note",
                     "memo", "recording", "clip", "gif", "sticker", "song", "playlist",
                     "selfie", "ticket", "receipt", "bill", "report", "presentation", "ppt",
                     "slides", "sheet", "spreadsheet", "zip", "paise", "paisa", "rupaye",
                     "rupay", "number", "address", "password", "otp", "detail", "code",
                     "pin", "amount", "cash"})
_SPOKEN_THING = re.compile(r"^(?:(?:a|an|the|my|ek)\s+)?(?:voice|audio|video)\s+"
                           r"(?:note|message|msg|memo|clip|recording|call)s?$")
# Hindi ends a request with these, and after the verb they are not a message: "ma ko
# message karna hai" sent "Hai", "... karo na" sent "Na".
_PARTICLES = frozenset({"na", "yaar", "yar", "please", "plz", "hai", "zara", "abhi",
                        "jaldi", "ji", "bhai", "now", "toh", "bas", "ab", "haan", "re"})


# ------------------------------------------------------------------ text

_ASKING = r"can\s+you|could\s+you|would\s+you|will\s+you"
# "I want YOU to text ma" asks for something. "I want to send flowers to mom" and "I
# need to text Rahul about the meeting" are the user thinking aloud, and with the "I
# want to" dropped as politeness they sent "Flowers" to Maa. So that stays in the
# sentence, and no form reads a sentence that starts with it.
_LEAD = re.compile(
    rf"^(?:(?:okay|ok|yes|yeah|yep|alright|hey|please|now|so|and|then|just|jev|kindly"
    rf"|{_ASKING}|i\s+(?:want|need)\s+you\s+to|go\s+ahead\s+and)\b[\s,.!]*)+", re.I)


def _tidy(utterance: str) -> str:
    """One line, straight apostrophes, and none of the politeness before the verb."""
    text = " ".join(utterance.replace("’", "'").split())
    lead = _LEAD.match(text)
    if lead is None:
        return text
    rest = text[lead.end():]
    # "Can you text ma that i'm late?" -- the question mark is on the request, and
    # left alone it arrived in the message as "I'm late?".
    if re.search(_ASKING, lead.group(0), re.I) and rest.endswith("?"):
        rest = rest[:-1].rstrip()
    return rest


def _letters(text: str, keep: str = "") -> str:
    """Lower-cased letters and digits, single-spaced. Emoji and punctuation go, so
    "Maa❤️" is "maa" here just as it is in chat.py; combining marks stay with their
    letter, or a Devanagari name falls apart."""
    out: list[str] = []
    for c in text.lower():
        kind = unicodedata.category(c)[0]
        attached = kind == "M" and bool(out) and out[-1] != " "
        out.append(c if (kind in "LN" or attached or c in keep) else " ")
    return " ".join("".join(out).split())


def _canonical(spoken: str) -> str:
    return "WhatsApp" if re.search(r"what|wats", spoken, re.I) else "Messages"


def _named(fragment: str) -> str:
    """The app a piece of command wrapper names, or ''."""
    for pattern in (_SPEC_RE, _APP_AS_VERB, _APP_BEFORE_NOUN):
        hit = pattern.search(fragment)
        if hit:
            return _canonical(hit.group(0))
    return ""


def _who(raw: str, limit: int = 3) -> str:
    """The name inside a captured stretch of sentence, or '' when it is not one."""
    text = re.sub(r"\s+for\s+me\b", " ", _SPEC_RE.sub(" ", raw.lower()))
    tokens = [t for t in _letters(re.sub(r"'s\b", "", text)).split() if t not in _WHO_NOISE]
    if not tokens or len(tokens) > limit:
        return ""
    if any(t in _NEVER_WHO for t in tokens):
        return ""
    return " ".join(tokens)


_KAR = r"(?:kar\s*do|karo|kar\s+dena|kar\s+dijiye|kar\s+de|kar)"
_BHEJ = r"(?:bhej\s*do|bhejo|bhej\s+dena|bhej\s+dijiye|bhej\s+de|bhejiye)"
# Sentences just outside the spec's list ("send the draft", "bhej dena") fell through to
# the general path, where "Send the text" was typed out to begin with. No words fit here.
_SEND_OBJECT = (r"(?:it|that|this|(?:(?:the|this|that|my)\s+)?(?:whatsapp\s+)?"
                r"(?:text\s+messages?|texts?|messages?|msgs?|draft|reply))")
_SEND_VERB = (r"(?:(?:hit|press|tap|click)\s+(?:the\s+)?send(?:\s+button)?"
              rf"|send(?:\s+{_SEND_OBJECT})?(?:\s+{_KAR})?"
              rf"|(?:(?:isko|ise|yeh|ye|message|msg)\s+)?(?:send\s+{_KAR}|{_BHEJ}))")
_SOON = (r"(?:please|now|right\s+now|right\s+away|already|then|quick|quickly|fast|asap"
         r"|abhi|jaldi|na|yaar|ji)")
_ENTER = r"(?:hit|press|tap)\s+(?:the\s+)?(?:enter|return)(?:\s+key)?"
# Whisper joins two quick commands with a full stop: "Type hello. Send it." was typed
# out whole, and the next "send" delivered "hello. Send it." to a person.
_SEND_TAIL = re.compile(
    r"(?:[\s,.;!?]*\b(?P<joined>and\s+then|and|then|aur\s+phir|aur|phir)\s+|\s*[,.;!?]+\s*)"
    rf"(?P<verb>{_SEND_VERB}|{_ENTER})(?:\s+{_SOON})*[\s.!?]*$", re.I)
_LEADING_MARK = re.compile(
    rf"^(?:{_ART}{_MSG_NOUN}\s+)?(?:saying|that\s+says|{_THAT})(?=[\s,:])[\s,:]*", re.I)
_LEADING_TO = re.compile(r"^to\s+", re.I)

# "message ma saying hi but don't send it" sent "Hi but don't send it".
_HOLD = (r"(?:(?:do\s+not|don'?t)\s+(?:send|(?:hit|press)\s+(?:send|enter))"
         r"(?:\s+(?:it|that|this|anything|the\s+(?:message|text)))?"
         r"(?:\s+(?:just\s+)?(?:yet|now|right\s+now|right\s+away))?"
         r"|without\s+sending(?:\s+(?:it|that|this))?"
         r"|(?:abhi\s+)?(?:bhejna|send\s+karna|send)\s+(?:mat|nahi|nahin)(?:\s+karna)?"
         r"|(?:abhi\s+)?mat\s+(?:bhejna|bhejo|bhej\s*do))")
_BUT = r"(?:but|and|par|lekin|magar|aur|just)"
_HOLD_TAIL = re.compile(
    rf"[\s,.;!?]*(?:\b{_BUT}\s+)?(?:\bplease\s+)?\b{_HOLD}[\s.!?]*$", re.I)
# Dictation has to be told apart from "type don't send it", which is words to type.
_HOLD_JOINED = re.compile(
    rf"(?:\s*[,.;!?]+\s*(?:{_BUT}\s+)?|\s+{_BUT}\s+)(?:please\s+)?{_HOLD}[\s.!?]*$", re.I)

# "Text Rahul I'm late. Never mind." sent "I'm late. Never mind." A message called off
# at its end is not one to send. It has to be its own clause -- after a stop, or after
# "no", "wait" -- so "don't cancel the trip" is still a message.
_WAIT = r"(?:no|nope|nahi|nahin|wait|actually|sorry|oh|um+|uh+)"
_CALLED_OFF = re.compile(
    rf"(?:[,.;!?]+\s*|\s+(?={_WAIT}\b))(?:{_WAIT}\b[\s,.;!?]*)*"
    r"(?:never\s*mind|forget\s+(?:it|that|about\s+it)|cancel(?:\s+(?:it|that|this))?"
    r"|scratch\s+that|leave\s+it|(?:rehne|rahne|jaane|jane)\s+do|chhodo|chodo"
    r"|chh?od\s+do)[\s.!?]*$", re.I)
# "Text Rahul, no, text Papa I'm late" sent Rahul "No, text Papa I'm late".
_RESTARTED = re.compile(
    rf"(?:^|[\s,.;!?])(?:no|nope|nahi|nahin|wait|sorry|actually|i\s+mean)\b[\s,.;!?]+"
    rf"(?:{_WAIT}\b[\s,.;!?]+)*(?:text|message|msg|tell|whatsapp|send|ask)\s+(?P<who>{_T})",
    re.I)
_NOT_WHO = frozenset({"not", "nahi", "nahin"})

_OBJECT_PRONOUN = re.compile(r"\b(?:it|them)[\s,]*$", re.I)
_OBJECT_NOUN = re.compile(r"(?:^|\s)(?P<verb>\S+)\s+(?:the|a|an|this|that|my|your|his|her"
                          r"|our|their)\s+\S+[\s,]*$", re.I)
_NOT_VERBS = frozenset({"on", "in", "at", "to", "for", "with", "from", "by", "of", "about",
                        "near", "after", "before", "into", "over", "under", "is", "am",
                        "are", "was", "and"})


def _without_whisper_stop(text: str) -> str:
    """Whisper ends every transcript with a full stop the user never said. A sentence
    with punctuation of its own was punctuated on purpose, and keeps its ending."""
    lone_stop = text.endswith(".") and not text.endswith("..")
    if lone_stop and not re.search(r"[.,;:!?]", text[:-1]):
        return text[:-1].rstrip()
    return text


def _belongs(before: str, cut: re.Match[str], dictated: bool) -> bool:
    """Is a trailing "and send it" the user's own sentence, and not a command?

    "message ma saying i will write the letter and send it" went out as "I will write
    the letter". A message is sent either way, so the cut is skipped when "it" has
    something to point back at just before the "and". Dictation keeps the spec's
    reading ("type the plan is ready and send it" sends) unless the clause before ends
    in "it" too: "type i will pack it and send it" sent half a sentence.

    A bare "and send" in a message is the user's too when what comes before asks the
    reader to do something: "fill the form and send" went out as "Fill the form", and
    "please sign and send" as "Please sign". "i reached and send" is still the command.
    """
    if not cut["joined"]:
        return False
    pointed = re.search(r"\b(?:it|that|this)\b", cut["verb"], re.I)
    bare = not dictated and cut["verb"].strip().lower() == "send"
    if not (pointed or bare):
        return False
    if pointed and _OBJECT_PRONOUN.search(before):
        return True
    if not dictated and re.match(r"please\b", before.strip(" ,"), re.I):
        return True
    noun = None if dictated else _OBJECT_NOUN.search(before)
    return noun is not None and noun["verb"].lower() not in _NOT_VERBS


# "call me on whatsapp", "i am on whatsapp", "are you on whatsapp": there the app is
# part of what is said, and lifted off as the app to use it sent "Call me" and "I am".
# So did a verb just before it: "don't call, text on whatsapp" sent "Don't call, text",
# and "i replied in messages" sent "I replied" -- through Messages.
_APP_IS_SAID = re.compile(
    r"(?:\b(?:am|is|are|was|were|be|been|being|me|us|him|her|them|it|this|that|online"
    r"|active|available|here|there|not|also|too|only|still|back|call|text|message|msg"
    r"|ping|reply|chat|talk|dm|\w+ed)"
    r"|'(?:m|s|re)"
    r"|\b(?:call|text|message|msg|ping|see|find|found|add|added|reach|contact|block"
    r"|blocked|dm|meet|catch)\s+(?:you|u)"
    r"|^(?:are|were|r)\s+(?:you|u))[\s,]*$", re.I)


def _lifted(text: str, lift: bool) -> tuple[str, str]:
    tail = _SPEC_TAIL.search(text) if lift else None
    if tail is None:
        return text, ""
    before = text[:tail.start()]
    # "... and send it on whatsapp" is the command, whatever "it" ends it.
    if _APP_IS_SAID.search(before) and not _SEND_TAIL.search(before):
        return text, ""
    return before, _canonical(tail.group(0))


def _spoken(raw: str, lift: bool, dictated: bool = False) -> tuple[str, str, bool]:
    """(the words, the app named after them, whether "and send" followed).

    A trailing "on whatsapp" is read as the app and not as part of the message: that
    is where people put it ("send hi to rudra on whatsapp"). `lift` is False once the
    wrapper has named an app, and then the tail is left alone as the user's words.
    """
    text, app = _lifted(raw.strip(" ,:-–—"), lift)
    cut = _SEND_TAIL.search(text)
    if cut and _belongs(text[:cut.start()], cut, dictated):
        cut = None
    if cut:                             # "... on whatsapp and send it", or the reverse
        text, late = _lifted(text[:cut.start()], lift and not app)
        app = app or late
    return _without_whisper_stop(text.strip(' "“”')), app, cut is not None


# ------------------------------------------------------------------ send

_YES = (r"(?:okay|ok|yes|yeah|yep|yup|ya|sure|alright|haan|han|ji|achha|acha|theek\s+hai"
        r"|now|ab|chalo|bas|please|just|go\s+ahead(?:\s+and)?)")
# Whisper repeats a short command it heard once: "Send it. Send it."
_SEND_ONLY = re.compile(
    rf"^(?:{_YES}\s+)*(?:{_SEND_VERB}(?:\s+{_SOON})*(?:\s+|$)){{1,3}}$")
# "Send." also comes back as "Sent."; "sent it on whatsapp", from elsewhere, is news.
_SENT_ONLY = re.compile(r"^sent(?:\s+it)?$")


def _send(text: str, scene: _Scene) -> ChatIntent | None:
    """A send only when the WHOLE sentence is one -- there is no body group in the
    pattern to capture words into, so there is nothing it could type."""
    flat = _letters(text, keep="'")
    named = _named(flat)
    # The app can sit mid-sentence: "send it on whatsapp please" was None.
    flat = " ".join(_SPEC_RE.sub(" ", flat, count=1).split())
    if scene.focused and not named and _SENT_ONLY.match(flat):
        return ChatIntent("send")
    if not _SEND_ONLY.match(flat) or not (scene.focused or named):
        return None
    return ChatIntent("send", app=named)


# ------------------------------------------------------------------ type

_TYPE = re.compile(r"^(?:type|write|likho|likh\s*do)(?:\s+out)?\b[\s,:]*(?P<body>.*)$", re.I)
_TYPE_FINAL = re.compile(r"^(?P<body>.+?)\s+(?:likho|likh\s*do)[\s.!?]*$", re.I)
# With a chat open, every "write ..." was dictation: "write a message to ma saying hi
# and send it" typed "a message to ma saying hi" into whichever chat was open and
# pressed send there. A sentence that names who it is for is never dictation.
_WRITTEN_FOR = re.compile(
    rf"^{_ART}(?:{_MSG_NOUN}|e-?mail|mail|letter|note|reply|response|answer|dm)\s+"
    r"(?:to|for)\s+\S", re.I)
_MESSAGE_FOR = re.compile(rf"^{_ART}{_MSG_NOUN}\s+(?:to|for)\s+\S", re.I)
# "write rudra a message saying call me", and "write me a message ..." which is nobody's.
_MESSAGE_WHO = re.compile(
    rf"^{_NAME}\s+(?:a|an|one|ek)\s+(?:(?:quick|short)\s+)?{_MSG_NOUN}{_MARK}\S", re.I)
# ... and with nothing after it: "type him a message and send it" sent "him a message".
_SOMEONE_A_MESSAGE = re.compile(
    rf"^{_NAME}\s+(?:a|an|one|ek)\s+(?:(?:quick|short)\s+)?{_MSG_NOUN}\b", re.I)
_WRITTEN_TO = re.compile(rf"^to\s+(?P<who>{_NAME}(?:\s+{_T}){{0,2}}?)[\s,:]+{_SAYING}", re.I)
# "type in rudra's chat saying hi" names the chat the words are for.
_IN_A_CHAT = re.compile(
    rf"^(?:in|into|on|to)\s+(?:(?:my|the)\s+)?"
    rf"(?:{_T}'s\s+{_NOUN}|{_NOUN}\s+(?:with|of|for)\s)", re.I)
# Hindi names who it is for first: "likho rudra ko ki main late hoon", "rudra ke liye
# message likho". Both were typed into the open chat, and the first was sent there.
_HI_WHO = re.compile(rf"^(?P<who>{_NAME})\s+(?:ko|ke\s+liye)\b[\s,]*", re.I)
# ... but "ghar ko ja raha hoon" is going home, and "main" is "I".
_HI_NOT_WHO = frozenset({"main", "mai", "mein", "hum", "ham", "tum", "tu", "aap", "ap",
                         "wo", "woh", "vo", "ye", "yeh", "sab", "kal", "aaj", "ghar"})


def _hindi_for(body: str) -> str:
    """The person a Hindi sentence starts by naming ("rudra ko ..."), or ''."""
    head = _HI_WHO.match(body)
    who = _who(head["who"]) if head else ""
    return "" if who in _HI_NOT_WHO else who


def _to_someone(body: str) -> bool:
    """"write to rudra that i am late" -- not "type to john, thanks for the gift"."""
    match = _WRITTEN_TO.match(body)
    return match is not None and bool(_who(match["who"]))


def _as_message(text: str, scene: _Scene) -> str:
    """"write a message to ma saying hi", with a verb the message forms know. Only
    with a marker: unmarked, "type a message to the team is ready" sent "Is ready"."""
    match = _TYPE.match(text)
    if match is None:
        return text
    body = re.sub(r"^up\s+", "", match["body"], flags=re.I)    # "type up a message to ..."
    addressed = _MESSAGE_FOR.match(body) and _MARK_RE.search(body)
    if addressed or _MESSAGE_WHO.match(body):
        return "send " + body
    if scene.focused and _to_someone(body):
        return "message " + _LEADING_TO.sub("", body)
    return text


# "type of thing, you know", "write that down": said about typing, not dictation. They
# typed "of thing, you know" and "down" into the open chat, and "... and send it" sent.
# "write up the report" and "write it off" are the same kind of sentence.
_NOT_DICTATION = re.compile(
    r"^(?:of\b(?!\s+course\b)|(?:up|off)\b|(?:(?:it|that|this|them|these|those|everything"
    r"|all\s+of\s+(?:it|that|this))\s+)?down\b)", re.I)
# ... and with a word or two in between, once "and send it" is off the end: "write the
# address down", "write it all down", "write that one down". "the server is down" and
# "calm down" are still somebody's words.
_PUT_DOWN = re.compile(
    r"^(?:it|that|this|them|these|those|everything|all|the|my|your|his|her|our|their|a"
    r"|an|some)(?:\s+[^\s']+)?\s+(?:down|off)[\s.!?]*$", re.I)
# "type hello to rudra and send it", "write to ma i am late and send it": a person named
# at either end is who the words are for, and the open chat may be anybody's.
_TO_WHO_HEAD = re.compile(rf"^to\s+(?P<who>{_NAME})(?:\s*[,:]\s*|\s+)(?=[^\s,:])", re.I)
_TO_WHO_END = re.compile(rf"^(?P<who>{_NAME}(?:\s+{_T}){{0,2}})[\s.!?]*$", re.I)
# After "to", these start what the user is going to do, not who it is for: "i want to
# go home" is not a message for somebody called Go Home.
_INFINITIVE = frozenset({
    "go", "come", "be", "do", "get", "see", "meet", "eat", "sleep", "leave", "reach",
    "talk", "call", "work", "stay", "wait", "pick", "buy", "pay", "have", "make", "take",
    "play", "watch", "start", "finish", "help", "check", "join", "bring", "know", "say",
    "tell", "send", "try", "visit", "drive", "study", "cook", "book", "give", "keep",
    "find", "sit", "run", "walk", "read", "write", "learn", "rest", "move", "catch",
    "attend", "hear", "ask", "speak", "share", "use", "fix", "clean", "wake"})


def _names_the_end(tail: str, prep: str) -> bool:
    """Is what follows a closing "to"/"for" a person? "for" needs the name's capital:
    "thanks for the help" is the user's, "hi for Rudra" is Rudra's."""
    hit = _TO_WHO_END.match(tail)
    who = _who(hit["who"]) if hit else ""
    if not who or who.split()[0] in _INFINITIVE:
        return False
    return prep == "to" or _proper(hit["who"].split()[-1])


def _for_someone(words: str) -> bool:
    head = _TO_WHO_HEAD.match(words)
    who = _who(head["who"]) if head else ""
    if who and who.split()[0] not in _INFINITIVE:
        return True
    return any(_names_the_end(words[at.end():], at["prep"].lower())
               for at in re.finditer(r"\b(?P<prep>to|for)\s+", words, re.I))


def _says_nothing(words: str) -> bool:
    """"the message", "a reply", "what i said", "this": they stand for a message
    without being one, and typed and sent they were the original bug again."""
    said = _letters(words)
    return bool(_PLACEHOLDER.match(said)) or said in _POINTERS


def _typed(raw: str) -> tuple[str, bool]:
    """(the words as the user said them, whether to send). No capital is added:
    dictation into an open chat is the user's text, not a composed message."""
    text = raw.strip(" ,:")
    if _NOT_DICTATION.match(text):
        return "", False
    held = _HOLD_JOINED.search(text)
    if held:
        text = text[:held.start()]
    words, _, send = _spoken(_LEADING_MARK.sub("", text.strip(" ,:")), lift=False,
                             dictated=True)
    if _PUT_DOWN.match(words) or _CALLED_OFF.search(words) or _says_nothing(words):
        return "", False
    return (words if words.strip(" .,!?") else ""), send and held is None


def _names_someone(body: str) -> bool:
    """Does a "type ..." sentence say who the words are for? Then it is not dictation
    into whichever chat is open."""
    return bool(_WRITTEN_FOR.match(body) or _MESSAGE_WHO.match(body)
                or _SOMEONE_A_MESSAGE.match(body) or _IN_A_CHAT.match(body)
                or _to_someone(body) or _hindi_for(body))


def _type(text: str, scene: _Scene, pattern: re.Pattern[str] = _TYPE) -> ChatIntent | None:
    match = pattern.match(text)
    if match is None or not scene.focused:      # elsewhere, typing is the general path's
        return None
    if _names_someone(match["body"]):
        return None
    words, send = _typed(match["body"])
    if send and _for_someone(words):
        return None                     # typed, it stays visible; sent, it cannot come back
    return ChatIntent("type", text=words, send=send) if words else None


# ------------------------------------------------------------------ open

_APP = rf"(?:(?:my|the)\s+)?(?:{_WA}|{_IM})(?:\s+(?:app|web|desktop|business))?"
# "whatsapp kholo" is "open whatsapp". It came back as a chat with someone called "kholo".
_APP_OPEN = re.compile(rf"^(?:{_OPEN}\s+{_APP}|{_APP}(?:\s+ko)?\s+{_KHOLO})$")


# "open instagram messages", "open notion chat": with no possessive and no "with", the
# name may be an app's, so this is the one chat-word form `reserved` can veto.
_INBOX_FORM = re.compile(rf"^{_OPEN}\s+(?P<who>.+?)\s+{_NOUN}$")
_OPEN_FORMS = (
    # "open my chat with ma", "my conversation with govind"
    re.compile(rf"^(?:{_OPEN}\s+)?(?:(?:my|the|a|our)\s+)?{_NOUN}\s+"
               r"(?:with|of|for|from)\s+(?P<who>.+)$"),
    # "open chat rudra"
    re.compile(rf"^{_OPEN}\s+(?:(?:my|the)\s+)?(?:chat|conversation|convo|dm|group)\s+"
               r"(?P<who>.+)$"),
    re.compile(rf"^{_OPEN}\s+(?P<who>.+?)'s\s+{_NOUN}$"),
    re.compile(rf"^{_OPEN}\s+(?P<who>.+?)\s+group(?:\s+chat)?$"),
    _INBOX_FORM,
    re.compile(rf"^(?P<who>.+?)'s\s+{_NOUN}$"),
    re.compile(rf"^(?P<who>.+?)(?:\s+(?:ki|ka|ke\s+saa?th(?:\s+wali)?|ke|wali|wala))?"
               rf"\s+(?:chat|conversation|group|messages|message|dm)(?:\s+ko)?\s+{_KHOLO}$"),
)
_BARE_FORMS = (
    re.compile(r"^(?:open\s+up|open|go\s+to|goto|switch\s+to|jump\s+to)\s+(?P<who>.+)$"),
    re.compile(rf"^(?P<who>.+?)(?:\s+ko)?\s+{_KHOLO}$"),
)


def _a_thing(token: str) -> bool:
    return token in _THINGS or (token.endswith("s") and token[:-1] in _THINGS)


def _taken(who: str, reserved: frozenset[str]) -> bool:
    """"open the photo" and "open settings" are not people, and neither is a name the
    caller already opens as an app or a site."""
    tokens = who.split()
    return who in reserved or any(
        t in reserved or t in _FURNITURE or _a_thing(t) for t in tokens)


def _open(text: str, scene: _Scene) -> ChatIntent | None:
    flat = _letters(text, keep="'")
    if _APP_OPEN.match(flat):                   # "open whatsapp" is an app open
        return None
    named = _named(flat) or scene.named
    flat = " ".join(_APP_BEFORE_NOUN.sub(" ", _SPEC_RE.sub(" ", flat)).split())
    for form in _OPEN_FORMS:
        match = form.match(flat)
        who = _who(match["who"], limit=4) if match else ""
        if who and not (form is _INBOX_FORM and _taken(who, scene.reserved)):
            return ChatIntent("open", who=who, app=named)
    # "open rudra" has no chat word in it. It is only a chat where chats are what is
    # on screen, and only when the name is not one "open" already means elsewhere.
    if not (scene.focused or named):
        return None
    for bare in _BARE_FORMS:
        match = bare.match(flat)
        who = _who(match["who"]) if match else ""
        if who and not _taken(who, scene.reserved):
            return ChatIntent("open", who=who, app=named)
    return None


# ------------------------------------------------------------------ message

@dataclass(frozen=True)
class _Form:
    """One way of saying "this person, these words"."""

    pattern: re.Pattern[str]
    needs_chat: bool = False        # too ordinary a phrase to act on outside a chat app
    drop_to: bool = False           # "tell rudra to call me" -> "Call me"
    literal: bool = False           # the body is whatever sat next to "send"
    plain: bool = False             # the name runs straight into the words, unmarked
    plain_min: int = 1              # fewest words an unmarked body may have
    hindi: bool = False             # verb-last grammar, and its particles


def _addressed_forms(prefix: str, plain_min: int, alone: bool,
                     **flags: bool) -> tuple[_Form, ...]:
    """Marked ("ma, how are you"), unmarked ("ma how are you"), and the name alone.

    Unmarked, only the first word can be trusted as the name. One stray word after it
    is likelier a surname than a message ("message kabir malhotra"), and the wrong guess
    sends "Malhotra" to a real person, so an unmarked body needs two words."""
    body = r"\S+\s+\S.*" if plain_min > 1 else r".+"
    forms = [
        _Form(re.compile(rf"^{prefix}(?P<who>.+?)(?P<mark>{_MARK})(?P<body>.+)$", re.I),
              **flags),
        _Form(re.compile(rf"^{prefix}(?P<who>{_NAME})(?:\s+{_SPEC})?\s+(?P<body>{body})$",
                         re.I), plain=True, plain_min=plain_min, **flags),
    ]
    if alone:
        forms.append(_Form(re.compile(rf"^{prefix}(?P<who>.+)$", re.I), **flags))
    return tuple(forms)


_SEND_NOUN = rf"(?:send|drop|shoot)\s+{_ART}{_MSG_NOUN}"
_HI_SEND = (r"(?:(?:bhejo|bhej\s*do|bhej\s+dena|bhej\s+de|bhejna(?:\s+hai)?|likho"
            r"|likh\s*do)\b)")
_HI_DO = (rf"(?:(?:karo|kar\s*do|kar\s+dena|kar\s+de|karna(?:\s+hai)?|kijiye|daalo"
          rf"|daal\s*do)\b|{_HI_SEND})")
_HI_TELL = r"(?:(?:bolo|bol\s*do|batao|bata\s*do|kaho|keh\s*do)\b)"
_KO = rf"^(?P<who>.+?)\s+ko\s+(?:{_SPEC}\s+)?"
# Hinglish names the app after the verb as well: "rudra ko message karo whatsapp pe ki
# main aa gaya" sent "Whatsapp pe ki main aa gaya".
_AFTER_VERB = rf"(?:\s+{_SPEC})?(?:(?P<mark>{_MARK})|\s+|$)"
_VERB_LAST = r"(?:\s+(?:na|yaar|please|ji))*[\s.!?]*$"

_FORMS = (
    # "send a message to rudra saying call me"
    *_addressed_forms(rf"{_SEND_NOUN}\s+(?:to|for)\s+", 2, alone=True),
    # "send a message saying call me to rudra"
    _Form(re.compile(rf"^{_SEND_NOUN}{_MARK}(?P<body>.+)\s+to\s+(?P<who>.+)$", re.I)),
    # "send rudra a message saying call me"
    _Form(re.compile(rf"^send\s+(?P<who>{_NAME}(?:\s+{_T}){{0,2}}?)\s+{_ART}{_MSG_NOUN}\b"
                     rf"(?:{_MARK}|\s+|$)(?P<body>.*)$", re.I)),
    # "send hi to rudra" -- split at the LAST "to": "send i want to go home to ma"
    _Form(re.compile(r"^send\s+(?P<body>.+)\s+to\s+(?P<who>.+)$", re.I), literal=True),
    # "message ma saying ...", "text govind that ...", "whatsapp rudra i am outside"
    *_addressed_forms(rf"(?:message|msg|text|{_WA}|i\s?message)\s+(?:to\s+)?", 2,
                      alone=True),
    *_addressed_forms(r"tell\s+", 1, alone=False, needs_chat=True, drop_to=True),
    # "ma ko message karo ki khana kha liya". After the verb an unmarked body needs two
    # words here too: one is usually a particle this module has not met yet.
    _Form(re.compile(rf"{_KO}{_ART}(?:{_MSG_NOUN}|{_WA})\s+{_HI_DO}{_AFTER_VERB}"
                     r"(?P<body>.*)$", re.I), hindi=True, plain_min=2),
    # "rudra ko whatsapp pe bhejo kal milte hain"
    _Form(re.compile(rf"{_KO}{_HI_SEND}{_AFTER_VERB}(?P<body>.+)$", re.I),
          hindi=True, plain_min=2, literal=True),
    _Form(re.compile(rf"{_KO}{_HI_TELL}{_AFTER_VERB}(?P<body>.+)$", re.I),
          hindi=True, plain_min=2, needs_chat=True),
    # Hindi puts the verb last: "ma ko hello bhej do". This is "send X to ma", so X may
    # be a thing or a pointer: "ma ko ye bhej do" sent "Ye", "ma ko photo bhej do" "Photo".
    _Form(re.compile(rf"{_KO}(?P<body>.+?)\s+{_HI_SEND}{_VERB_LAST}", re.I),
          hindi=True, literal=True),
    _Form(re.compile(rf"{_KO}(?P<body>.+?)\s+{_HI_TELL}{_VERB_LAST}", re.I),
          hindi=True, needs_chat=True),
)

# A capitalised word is only taken for a surname when a sentence plainly starts right
# after it. Whisper capitalises "Happy Diwali" as readily as "Malhotra".
_OPENS = re.compile(r"^(?:i|i'\w+|we|you|he|she|they|it|this|there|please|how|what|where"
                    r"|when|why)$", re.I)
_TOPIC = re.compile(r"^(?:about|regarding|concerning)\b", re.I)
# Hindi puts "about" last: "Rahul ko message karo meeting ke baare mein".
_TOPIC_LAST = re.compile(r"\b(?:ke|ki)\s+ba+re\s+m(?:ein|ain|en|e)[\s.!?]*$", re.I)
# How or when, and not what: "Text Rahul for me" sent "For me", "text rahul right now"
# sent "Right now".
_ASIDE = re.compile(
    r"^(?:(?:for\s+me|as\s+well|too|also|again|if\s+possible|if\s+you\s+can|right\s+now"
    r"|right\s+away|now|asap|please|quickly|jaldi|abhi)\b[\s,.!?]*)+$", re.I)
# "text to speech is not working", "text size is too small": the "name" was the subject
# of a sentence about something else ("are you coming" is still a message). And "message
# ma and rudra" sent "And rudra": after a name, "and" starts a step or another name, and
# "or" another name. "text Rahul when you get there", "text Rahul when it's done",
# "message mom from your phone": said to somebody in the room -- when or how THEY should
# do it, not the words.
_NOT_A_MESSAGE = re.compile(
    r"^(?:(?:and|or|then|aur|ya|phir)\b"
    r"|(?:when|whenever|once|after|before|if|until|till|as\s+soon\s+as)"
    r"\s+(?:you|u|he|she|they|we|i|it|that|this)\b"
    r"|from\s+(?:your|his|her|their)\b"
    r"|(?:is|are|was|were)\s+(?:not|too|so|very|still|being|broken|down|working)\b"
    r"|(?:isn't|aren't|wasn't|weren't|doesn't|does\s+not)\s+(?:work|open|load|send|show))",
    re.I)
# "Rahul ko message karna hai ..." is "I have to message Rahul": the chat, never the words.
_NEED_TO = re.compile(r"\b(?:karna|bhejna|likhna)\b(?!\s+(?:mat|nahi|nahin)\b)", re.I)
_LISTED = re.compile(r"^(?P<names>[^.!?]*?)[\s,]*\b(?:saying|that\s+says|ki)\b", re.I)


def _is_thing(said: str) -> bool:
    """"send the link to rudra", "send 500 rupees to rudra", "send pictures to ma",
    "send photo of the cat to ma", "papa ko 500 bhej do": an object, not words."""
    tokens = said.split()
    return bool(tokens) and (tokens[0] in _DETERMINERS or _a_thing(tokens[0])
                             or tokens[0].isdigit() or _a_thing(tokens[-1])
                             or bool(_SPOKEN_THING.match(said)))


def _an_object(said: str) -> bool:
    """"text rahul the location", "whatsapp rahul my location": after the name comes a
    thing to send and not a sentence. Narrower than _is_thing, because "text rahul the
    meeting is at five" is a message."""
    tokens = said.split()
    return ((1 < len(tokens) <= 4 and tokens[0] in _DETERMINERS and _a_thing(tokens[-1]))
            or bool(_SPOKEN_THING.match(said)))


def _second_thoughts(words: str, who: str) -> bool:
    """Called off at the end, or started again for somebody else: "I'm late. Never
    mind.", "no, text Papa I'm late", "hi. Wait, not Rahul, Papa"."""
    if _CALLED_OFF.search(words):
        return True
    if any(_who(again["who"]) for again in _RESTARTED.finditer(words)):
        return True
    said, size = _letters(words).split(), len(who.split())
    return any(t in _NOT_WHO and " ".join(said[i + 1:i + 1 + size]) == who
               for i, t in enumerate(said))


def _proper(token: str) -> bool:
    name = _letters(token)
    return (token[:1].isupper() and not token.isupper() and bool(name)
            and name not in _NEVER_WHO and not _OPENS.match(token))


_GROUP_TAIL = re.compile(r"^(?P<rest>(?:\S+\s+){1,2})group(?:\s+chat)?[\s.!?]*$", re.I)


def _full_name(match: re.Match[str], text: str, form: _Form) -> tuple[str, str]:
    """(name, body) for an unmarked form, with a capitalised surname moved across:
    "Text Kabir Malhotra I am late" sent "Malhotra I am late" to Kabir."""
    who, body = match["who"], match["body"]
    if text[match.end("who"):match.start("body")].strip():
        return who, body                # an app is named in between
    group = _GROUP_TAIL.match(body)
    if group and not set(_letters(group["rest"]).split()) & _DETERMINERS:
        return f"{who} {body}", ""      # "text college boys group" sent "Boys group"
    tokens = body.split()
    for count in (1, 2):
        if len(tokens) - count < form.plain_min or not _proper(tokens[count - 1]):
            break
        if _OPENS.match(tokens[count]):
            return f"{who} {' '.join(tokens[:count])}", " ".join(tokens[count:])
    return who, body


def _several(mark: str, body: str) -> bool:
    """"message ma, papa, and didi saying dinner is ready": the comma was a list, and
    read as the marker it sent "Papa, and didi saying dinner is ready" to ma."""
    listed = _LISTED.match(body) if "," in mark else None
    if listed is None or not re.search(r",|\b(?:and|aur)\b", listed["names"], re.I):
        return False
    names = [t for t in _letters(listed["names"]).split() if t not in ("and", "aur")]
    return 0 < len(names) <= 4 and not any(t in _NEVER_WHO for t in names)


def _compose(who: str, named: str, words: str, scene: _Scene, send: bool = True,
             literal: bool = False, capital: bool = True,
             asked: bool = False) -> ChatIntent | None:
    """A message when there are words to send; with a person and no words, only the
    chat to open. `asked` is True once the sentence has asked for this chat by name."""
    chat = ChatIntent("open", who=who, app=named)
    said = _letters(words)
    if _PLACEHOLDER.match(said):
        return chat
    if said in _POINTERS:
        # "send it to rudra" said over a Finder window is about a file. Only where
        # chats are in play does "it" mean the draft, and even then nothing is sent.
        in_chat = bool(scene.focused or named or asked)
        return chat if (in_chat or not literal) else None
    if literal and _is_thing(said):
        return chat if asked else None
    return ChatIntent("message", who=who, app=named,
                      text=words[:1].upper() + words[1:] if capital else words, send=send)


def _body(words: str, mark: str, form: _Form) -> str:
    """The words worth sending, or "" when what followed the name was not a message."""
    said = _letters(words).split()
    # "text rudra about the meeting", and whisper's "Text Rahul, about the meeting":
    # a comma is not "saying", and does not turn a topic into the words.
    unsaid = not re.search(r"[a-z]", mark, re.I)
    if unsaid and (_TOPIC.match(words) or _TOPIC_LAST.search(words) or _ASIDE.match(words)):
        return ""
    # "Message ma, send it" is two commands. "tell rudra to send it" is for Rudra to do.
    if "," in mark and _SEND_ONLY.match(" ".join(said)):
        return ""
    if not mark and len(said) < form.plain_min:
        return ""
    if form.hindi and all(t in _PARTICLES for t in said):
        return ""
    return _LEADING_TO.sub("", words) if form.drop_to else words


def _read(match: re.Match[str], text: str, form: _Form, scene: _Scene) -> ChatIntent | None:
    fields = match.groupdict()
    mark = fields.get("mark") or ""
    name, raw = _full_name(match, text, form) if form.plain else (
        match["who"], fields.get("body") or "")
    who = _who(name)
    if not who or _several(mark, raw) or (form.plain and _NOT_A_MESSAGE.match(raw)):
        return None
    if "body" not in fields and _an_object(" ".join(_letters(name).split()[1:])):
        return None                     # "text rahul the link", read whole as a name
    wrapper = text[:match.start("body")] if raw else text
    named = _named(wrapper) or _named(match["who"]) or scene.named
    words, tail_app, _ = _spoken(raw, lift=not named)
    named = named or tail_app
    if form.needs_chat and not (scene.focused or named):
        return None
    if _second_thoughts(words, who):
        return None
    if not re.search(r"[a-z]", mark, re.I) and _an_object(_letters(words)):
        return None                     # "WhatsApp Rahul the PDF" sent "The PDF"
    if form.hindi and _NEED_TO.search(_around(match, text)):
        words = ""
    return _compose(who, named, _body(words, mark, form), scene, literal=form.literal)


def _around(match: re.Match[str], text: str) -> str:
    """The sentence without its body: where a Hindi verb sits, before or after it."""
    return f"{text[:match.start('body')]} {text[match.end('body'):]}"


def _message(text: str, scene: _Scene) -> ChatIntent | None:
    for form in _FORMS:
        match = form.pattern.match(text)
        intent = _read(match, text, form, scene) if match else None
        if intent is not None:
            return intent
    return None


# ------------------------------------------------------------------ two steps

_THEN = re.compile(
    rf"^(?P<left>.+?)[\s,]+(?:and\s+then|and|then|aur\s+phir|aur|phir)\s+"
    rf"(?P<right>(?:{_ACT}|{_OPEN})\b.*)$", re.I)
_SAY = re.compile(
    r"^(?P<verb>send|text|message|msg|tell|ask|say|bolo|bhejo)\b"
    rf"(?P<to>\s+to)?(?P<them>\s+(?:her|him|them))?(?P<noun>\s+{_ART}{_MSG_NOUN}\b)?"
    rf"(?:(?P<mark>{_MARK})|\s+|$)(?P<body>.*)$", re.I)


_TO_THEM = re.compile(r"\s+to\s+(?:her|him|them)$", re.I)


# After "tell" and "ask" comes who is told: "... and ask rudra where he is" sent Ma
# "Rudra where he is". These start what is asked instead.
_NOT_TOLD = frozenset({"if", "whether", "about"})


def _told(chat: ChatIntent, say: re.Match[str]) -> str | None:
    """The words after the verb, or None when somebody else is the one told.

    After "tell" and "ask" comes who is told, and so it does after "say to", and in
    Hindi "bhejo papa ko": "... and say to rudra i'm late" sent Ma "Rudra i'm late", and
    "... and bhejo papa ko hi" sent Rudra "Papa ko hi"."""
    body = say["body"]
    if say["them"] or say["noun"] or say["mark"]:
        return body
    hindi = _hindi_for(body)
    if hindi:
        return body[_HI_WHO.match(body).end():] if hindi == chat.who else None
    if say["verb"].lower() not in ("tell", "ask") and not say["to"]:
        return body
    words = body.split()
    size = len(chat.who.split())
    if _who(" ".join(words[:size])) == chat.who:
        return " ".join(words[size:])   # "... and ask ma where papa is"
    first = words[0] if words else ""
    if first and first.lower() not in _NOT_TOLD and _who(first):
        return None
    return f"to {body}" if say["to"] else body     # "... and say to be honest i forgot"


def _then_say(chat: ChatIntent, right: str, scene: _Scene) -> ChatIntent | None:
    """The second half of "open my chat with ma and ...", once the chat is known."""
    # Read as if a chat app were in front, where "tell rudra i'm late" is a message to
    # Rudra. Over Finder it was not, and the words went to Ma as "Rudra i'm late".
    inside = replace(scene, focused=scene.focused or CHAT_APPS[0])
    other = _message(_as_message(right, inside), inside)
    if other is not None:
        # Naming somebody again is fine when it is the same somebody. Anyone else is
        # two people in one sentence, and "... and message rudra" sent "Rudra" to ma.
        return replace(other, app=other.app or chat.app) if other.who == chat.who else None
    write = _TYPE.match(right)
    if write:                           # "type" never sends by itself, here either
        if _names_someone(write["body"]):
            return None
        words, send = _typed(write["body"])
        if send and _for_someone(words):
            return None
        return _compose(chat.who, chat.app, words, scene, send=send, capital=False,
                        asked=True)
    say = _SAY.match(right)
    told = _told(chat, say) if say else None
    if told is None:
        return None
    words, tail_app, _ = _spoken(told, lift=not chat.app)
    if _second_thoughts(words, chat.who):
        return None
    if say["verb"].lower() in ("tell", "ask"):
        words = _LEADING_TO.sub("", words)
    if say["verb"].lower() == "send":   # "... and send it to her" sent "It to her"
        words = _TO_THEM.sub("", words)
    # "... and send her the photo": with no "a text" and no "saying", what follows
    # "send" may be a thing, as in "send the photo to ma". And "... and send it" is the
    # chat alone: nobody has seen what its compose box holds, so nothing is sent blind.
    literal = say["verb"].lower() in ("send", "bhejo") and not (say["noun"] or say["mark"])
    return _compose(chat.who, chat.app or tail_app, words, scene, literal=literal,
                    asked=True)


def _first_step(left: str, scene: _Scene) -> ChatIntent | None:
    """What "X and then ..." opens first: a chat, or (who == "") just the app."""
    flat = _letters(left, keep="'")
    if _APP_OPEN.match(flat):           # "open whatsapp and text ma that i am late"
        return ChatIntent("open", app=_canonical(flat))
    # "text ma and tell her i am late": naming somebody without any words opens their
    # chat as well. Read as one step, it sent "And tell her i am late".
    chat = _open(left, scene) or _message(left, scene)
    return chat if chat is not None and chat.op == "open" else None


def _second_step(first: ChatIntent, right: str, scene: _Scene) -> ChatIntent | None:
    if first.who:
        return _then_say(first, right, replace(scene, named=scene.named or first.app))
    inside = replace(scene, named=first.app)
    return _message(_as_message(right, inside), inside) or _open(right, inside)


# ------------------------------------------------------------------ entry

def _person(text: str, scene: _Scene) -> ChatIntent | None:
    """Everything that names a person: open, message, and the two in one sentence."""
    other = _ELSEWHERE.search(text)
    if other and not _MARK_RE.search(text[:other.start()]):
        return None                     # after "saying", "on instagram" is the user's words
    head = _SPEC_HEAD.match(text)
    if head:                            # "whatsapp pe ma ki chat kholo"
        scene = replace(scene, named=_canonical(head.group(0)))
        text = text[head.end():]
    steps = _THEN.match(text)
    first = _first_step(steps["left"], scene) if steps else None
    if steps and first:
        # Committed to two steps. A second half that cannot be read refuses the whole
        # sentence; read whole instead, "open my chat with ma and open youtube" came
        # back as a chat called "ma and open youtube".
        return _second_step(first, steps["right"], scene)
    return _message(text, scene) or _open(text, scene)


def _addressed(text: str, scene: _Scene) -> ChatIntent | None:
    """_person(), minus a closing "but don't send it" -- which makes it a draft."""
    held = _HOLD_TAIL.search(text)
    if held is None:
        return _person(text, scene)
    intent = _person(text[:held.start()], scene) if held.start() else None
    return replace(intent, send=False) if intent is not None else None


def parse(utterance: str, focused_app: str = "",
          reserved: frozenset[str] = frozenset()) -> ChatIntent | None:
    """The messaging intent in `utterance`, or None to leave it to the general path.

    `reserved` is lower-cased names that "open X" already means something for (apps,
    sites); it vetoes bare "open rudra" and "open notion messages", never a sentence
    that says whose chat it is. `app` comes back "" when none was named and the focus is
    not a chat app. A "message" with send=False is a draft: type it there and stop.
    """
    if not isinstance(utterance, str) or len(utterance) > MAX_CHARS:
        return None
    if len(utterance.split()) > MAX_WORDS:
        return None
    text = _tidy(utterance)
    if not text:
        return None
    focused = focused_app if focused_app in CHAT_APPS else ""
    scene = _Scene(focused=focused, reserved=reserved)
    intent = (_send(text, scene) or _type(text, scene)
              or _addressed(_as_message(text, scene), scene)
              or _type(text, scene, _TYPE_FINAL))
    if intent is None:
        return None
    if not intent.app and focused_app.casefold() in _OTHER_MESSENGERS:
        return None
    return replace(intent, app=intent.app or scene.focused)
