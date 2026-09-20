"""Route a fixed set of utterances through each Jev model; score + time them.

Usage: uv run python bench.py ~typesafe/jev-latest
"""
import statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from jev_voice.brain import Brain

FRONT = "Google Chrome"
# (utterance, expected action, expected {arg: value} subset)
CASES = [
    ("open chrome",                      "open_app",      {"app": "Google Chrome"}),
    ("switch to cursor",                 "open_app",      {"app": "Cursor"}),
    ("go to youtube",                    "open_website",  {"site": "youtube"}),
    ("pull up gmail",                    "open_website",  {"site": "gmail"}),
    ("search youtube for lofi hip hop",  "web_search",    {"engine": "youtube", "query": "lofi hip hop"}),
    ("google best ramen near me",        "web_search",    {"engine": "google", "query": "best ramen near me"}),
    ("type hello world and hit enter",   "type_text",     {"text": "hello world", "submit": True}),
    ("write good morning everyone",      "type_text",     {"text": "good morning everyone", "submit": False}),
    ("close this tab",                   "shortcut",      {"shortcut": "close_tab_or_window"}),
    ("select all and copy",              "shortcut",      {}),
    ("undo",                             "shortcut",      {"shortcut": "undo"}),
    ("scroll down a lot",                "scroll",        {"direction": "down", "amount": "a_lot"}),
    ("go to the top",                    "scroll",        {"direction": "top"}),
    ("volume up",                        "volume",        {"op": "up"}),
    ("mute",                             "volume",        {"op": "mute"}),
    ("pause the music",                  "media",         {"op": "play_pause"}),
    ("next song",                        "media",         {"op": "next"}),
    ("take a screenshot",                "screenshot",    {}),
    ("open my downloads",                "open_folder",   {"folder": "downloads"}),
    ("lock the screen",                  "system",        {"op": "lock"}),
    ("toggle dark mode",                 "system",        {"op": "toggle_dark_mode"}),
    ("new note called groceries",        "new_item",      {"title": "groceries"}),
    ("stop listening",                   "stop",          {}),
    # regressions from the 2026-09-20 live session
    ("quit spotify",                     "close_app",     {"app": "Spotify"}),
    ("close chrome",                     "close_app",     {"app": "Google Chrome"}),
    ("play until i found you on spotify","play_track",    {"query": "until i found you"}),
    ("play taylor swift",                "play_track",    {"query": "taylor swift"}),
    ("pause",                            "media",         {"op": "play_pause"}),
    ("open spotify",                     "open_app",      {"app": "Spotify"}),
]
# utterances that must NOT be treated as in_app (no app named in the words)
NO_IN_APP = {"type hello world and hit enter", "write good morning everyone",
             "scroll down a lot", "close this tab", "undo", "pause", "quit spotify"}

def run(model: str):
    b = Brain(model=model)
    b.evaluate("open chrome", front=FRONT)  # warm TLS + prompt cache

    def one(case):
        u, want_act, want_args = case
        t0 = time.perf_counter()
        try:
            p = b.evaluate(u, front=FRONT)
        except Exception as e:
            return u, False, [f"ERROR {type(e).__name__}: {e}"], (time.perf_counter()-t0)*1000
        ms = (time.perf_counter() - t0) * 1000
        bad = []
        if p.action != want_act:
            bad.append(f"action={p.action} want {want_act}")
        for k, v in want_args.items():
            got = p.args.get(k)
            if isinstance(v, str) and isinstance(got, str):
                if got.strip().casefold() != v.strip().casefold():
                    bad.append(f"{k}={got!r} want {v!r}")
            elif got != v:
                bad.append(f"{k}={got!r} want {v!r}")
        if u in NO_IN_APP and p.args.get("in_app"):
            bad.append(f"spurious in_app={p.args['in_app']!r}")
        return u, not bad, bad, ms

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one, CASES))

    # latency measured serially on a few, since parallel inflates it
    serial = []
    for u, _, _, _ in results[:5]:
        t0 = time.perf_counter(); b.evaluate(u, front=FRONT); serial.append((time.perf_counter()-t0)*1000)

    ok = sum(1 for _, good, _, _ in results if good)
    print(f"\n{'='*72}\n{model}\n  score {ok}/{len(CASES)}   "
          f"serial latency med {statistics.median(serial):.0f}ms  "
          f"min {min(serial):.0f}ms  max {max(serial):.0f}ms")
    for u, good, bad, ms in results:
        if not good:
            print(f"    ✗ {u!r}: {'; '.join(bad)}")
    return model, ok, statistics.median(serial)

if __name__ == "__main__":
    out = []
    for m in sys.argv[1:]:
        try:
            out.append(run(m))
        except Exception as e:
            print(f"\n{m}: FAILED {type(e).__name__}: {e}")
    print(f"\n{'='*72}\nSUMMARY (score, median latency)")
    for m, ok, ms in sorted(out, key=lambda r: (-r[1], r[2])):
        print(f"  {m:<42} {ok}/{len(CASES)}  {ms:.0f}ms")
