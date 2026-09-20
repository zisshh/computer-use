"""In-process Apple Events, instead of shelling out to `osascript`.

Every AppleScript in this project used to cost a process spawn: 110-190ms measured,
about 60ms of which is fork/exec that does no work. The same events sent from inside
this process cost 6-9ms. That 17x is what makes a per-utterance screen snapshot
affordable at all -- see context.ContextWatcher.

The public surface deliberately mirrors the old helper (`run` raises RuntimeError with
the script's own error text, returns stdout-shaped text), so callers did not change.
"""
from __future__ import annotations

import subprocess
import threading

_LOCK = threading.Lock()          # AppleScript execution is not re-entrant
_CACHE: dict[str, object] = {}    # source -> compiled NSAppleScript
_CACHE_MAX = 256

try:
    from Foundation import NSAppleScript, NSURL
    from AppKit import NSWorkspace, NSWorkspaceOpenConfiguration

    AVAILABLE = True
except Exception:                 # pragma: no cover - pyobjc missing
    AVAILABLE = False

_TYPE_LIST = 1818850164           # 'list'


def _text(desc) -> str:
    """Render a descriptor the way `osascript` prints it."""
    if desc is None:
        return ""
    if desc.descriptorType() == _TYPE_LIST:
        return ", ".join(_text(desc.descriptorAtIndex_(i + 1))
                         for i in range(desc.numberOfItems()))
    value = desc.stringValue()
    return "" if value is None else str(value)


def with_timeout(script: str, seconds: int) -> str:
    """Bound an Apple Event that can hang.

    `execute javascript` against a tab that is still loading blocks until the default
    60s Apple Event timeout -- in-process that stalls the worker thread, where a
    subprocess could simply be killed. Anything touching a web page gets a bound.
    """
    return f"with timeout of {seconds} seconds\n{script}\nend timeout"


def _compiled(source: str):
    script = _CACHE.get(source)
    if script is None:
        script = NSAppleScript.alloc().initWithSource_(source)
        ok, _ = script.compileAndReturnError_(None)
        if not ok:
            return None
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[source] = script
    return script


def run(script: str, timeout: int | None = None) -> str:
    """Execute AppleScript and return its result as text. Raises RuntimeError on error."""
    source = with_timeout(script, timeout) if timeout else script
    if not AVAILABLE:
        return _subprocess(source)
    with _LOCK:
        compiled = _compiled(source)
        if compiled is None:
            return _subprocess(source)      # a compile quirk we do not want to mask
        desc, err = compiled.executeAndReturnError_(None)
    if desc is None:
        message = ""
        if err is not None:
            message = str(err.get("NSAppleScriptErrorMessage", "") or err)
        raise RuntimeError(message or "AppleScript failed")
    return _text(desc).strip()


def _subprocess(script: str) -> str:
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip()


# ---------------------------------------------------------------- URLs

def open_url(url: str, activate: bool = True) -> bool:
    """Hand a URL to its registered app without spawning `open`.

    `activate=False` is the `open -g` equivalent: Discord and Notion navigate without
    stealing focus, which matters when the user is mid-sentence in another window.
    """
    if not AVAILABLE:
        args = ["open"] + ([] if activate else ["-g"]) + [url]
        return subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL) is not None
    ns = NSURL.URLWithString_(url)
    if ns is None:
        return False
    workspace = NSWorkspace.sharedWorkspace()
    try:
        config = NSWorkspaceOpenConfiguration.configuration()
        config.setActivates_(activate)
        workspace.openURL_configuration_completionHandler_(ns, config, None)
        return True
    except Exception:
        return bool(workspace.openURL_(ns))



def open_url_in(app: str, url: str, activate: bool = True) -> bool:
    """Open `url` in a named application, bypassing the default handler.

    This machine's default http handler is Velja, a URL router -- so handing it a URL
    means the page lands wherever Velja's rules say, which is not necessarily the
    browser the user is looking at. Naming the browser removes the guess.
    """
    if not AVAILABLE:
        return subprocess.Popen(["open", "-a", app, url], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL) is not None
    ns = NSURL.URLWithString_(url)
    workspace = NSWorkspace.sharedWorkspace()
    bundle = workspace.URLForApplicationWithBundleIdentifier_(app)
    if bundle is None:
        for base in ("/Applications/", "/System/Applications/"):
            candidate = NSURL.fileURLWithPath_(f"{base}{app}.app")
            if candidate and candidate.checkResourceIsReachableAndReturnError_(None)[0]:
                bundle = candidate
                break
    if ns is None or bundle is None:
        return False
    config = NSWorkspaceOpenConfiguration.configuration()
    config.setActivates_(activate)
    workspace.openURLs_withApplicationAtURL_configuration_completionHandler_(
        [ns], bundle, config, None)
    return True

# ---------------------------------------------------------------- MediaRemote

_MR: dict[str, object] = {}


def _mediaremote():
    """Lazily load the private MediaRemote framework. ~8ms once, then 0.03ms a call."""
    if "loaded" not in _MR:
        _MR["loaded"] = False
        try:
            import ctypes

            path = "/System/Library/PrivateFrameworks/MediaRemote.framework/MediaRemote"
            lib = ctypes.cdll.LoadLibrary(path)
            lib.MRMediaRemoteSendCommand.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.MRMediaRemoteSendCommand.restype = ctypes.c_bool
            _MR["lib"] = lib
            try:
                import objc
                from Foundation import NSBundle

                NSBundle.bundleWithPath_(
                    "/System/Library/PrivateFrameworks/MediaRemote.framework").load()
                _MR["request"] = objc.lookUpClass("MRNowPlayingRequest")
            except Exception:
                _MR["request"] = None
            _MR["loaded"] = True
        except Exception:
            pass
    return _MR if _MR.get("loaded") else None


# MediaRemote command codes.
MR_PLAY, MR_PAUSE, MR_TOGGLE, MR_NEXT, MR_PREVIOUS = 0, 1, 2, 4, 5


def media_command(code: int) -> bool:
    """Send a transport command to whichever app macOS says owns playback.

    This replaces synthesising an NX_KEYTYPE_PLAY HID event: same routing, but no
    fake keypress that a focused app might swallow, and 0.3ms instead of ~5ms.
    """
    mr = _mediaremote()
    if mr is None:
        return False
    try:
        return bool(mr["lib"].MRMediaRemoteSendCommand(code, None))
    except Exception:
        return False


def now_playing_owner() -> str | None:
    """Bundle id of the app that currently owns Now Playing, or None.

    Only the owner is readable -- the track metadata behind it comes back empty
    (entitlement-gated), so titles must be asked of the owning app itself.
    """
    mr = _mediaremote()
    if mr is None or mr.get("request") is None:
        return None
    try:
        path = mr["request"].localNowPlayingPlayerPath()
        client = path.client() if path is not None else None
        return str(client.bundleIdentifier()) if client is not None else None
    except Exception:
        return None
