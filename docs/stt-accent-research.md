# STT accent research: decision memo

2026-09-21 - jev-voice - Apple M1 Max 64 GB, macOS 27.0 (26A428)

Inputs: five research lanes (apple, parakeet, whisper-variants, hosted, eval). Each was re-checked by a skeptic who re-ran the local probes on this Mac and re-fetched the cited sources. Refuted and unverifiable claims are dropped or flagged here.

Caveat on every accuracy number below: no clip in any lane was the real user. Local accuracy tests used Apple TTS voices (Rishi, Aman, Tara) or macOS `say`; public numbers are read speech from other speakers. They show where to look. They cannot rank engines for this voice. Latency, output script, API shape and integration facts were measured on this machine and are solid.

## 1. Bottom line

Do not replace whisper small.en yet; augment it. No whisper variant, bigger whisper or Indian-tuned whisper meets script + latency + English-verb accuracy together, and nothing surveyed is proven on the user's own voice. The best-supported move is a two-engine front end. Keep small.en + bias prompt, because it wins on the short English commands that are the actual complaint (8/9 vs 5/9 on type hello / type it / deafen me). Run Apple's DictationTranscriber (en-IN) in a Swift sidecar in parallel, because it wins on Hinglish sentences and Indian names (27/36 vs 22/36 exact overall), emits Latin-script Hinglish natively, costs 77-92 ms median, is already installed, and needs no TCC prompt or app bundle. Arbitrate with the verb lexicon and entity catalog the project already has. Before building it: fix the bias-prompt budget bug (free, and the prompt is the strongest measured lever), and stand up a passive real-voice eval of at least 300 labeled utterances. Bench Apple Dictation, Qwen3-ASR-0.6B (the only local model with real-speaker Indian-accent evidence, 11.1% vs 17.6% WER for whisper-small, but it leaks Devanagari) and Sarvam saaras:v4 translit (hosted fallback only) against the incumbent on that set. "Accent-proof" most likely ends in adapting small.en to this speaker on the same recordings; that is phase 2 because it needs that data first.

## 2. Ranked shortlist

| # | Option | Local / hosted | Indian-accent evidence | Expected latency on M1 Max | Vocabulary biasing | Integration effort | Confidence |
|---|---|---|---|---|---|---|---|
| 0 | Control: whisper small.en + bias prompt | Local (Metal) | TTS only: 22/36 exact, 8/9 on short commands. Public proxy: multilingual whisper-small 17.6% WER on Common Voice 24 Indian-accent (n=511) | Measured 133-155 ms. 51-68 ms with `audio_ctx` 512-768 (TTS only, unproven on real voice) | Initial prompt, 223 usable tokens, tail kept. Strongest measured lever: 32% -> 74% keyword recall; 8/36 -> 22/36 exact | None | Baseline |
| 1 | Apple DictationTranscriber en-IN + contextualStrings, in parallel with small.en | Local, OS model | None published. TTS only: 27/36 vs whisper 22/36; wins Hinglish and names, loses short commands 5/9 vs 8/9 ('type it' 0/3 voices). One HN anecdote. Only published proxy for this model family (legacy on-device engine) is about 2x worse than Whisper Small on LibriSpeech | Measured: median 77-92 ms, p95 207, max 335 (one 3.1 s clip reliably 230-335 ms). Streaming variant 29-43 ms post-speech | contextualStrings, max 100 phrases of 1-2 words. Weak: changed 4/36 transcripts; most names were already right unbiased. Custom LM: zero effect in two attempts | Medium: ~150-line Swift sidecar (plain `swiftc`, no bundle, no TCC for buffer input), Python client, arbiter. PyObjC cannot reach these APIs | Medium on latency, script, integration (reproduced here). Low on accuracy |
| 2 | Qwen3-ASR-0.6B-8bit via mlx-audio + hotwords | Local (MLX, GPU) | Best independent evidence: CV24 Indian-accent 11.1% WER vs whisper-small 17.6% (1.7B: 7.8%). Vendor accented-English set 16.62 vs whisper-large-v3 21.30 | Measured under load: p50 84-168 ms, p95 120-241; 109-201 ms median with 50 hotwords. 1.7B is 222-397 ms with 50 hotwords (over budget) | `hotwords=[...]` / `context`, folded into the system prompt. Fixed 3/3 entity errors on TTS (n=5). No published efficacy numbers | Install is trivial (`pip install mlx-audio`, in-process). Meeting the Latin-script requirement is not: Devanagari leaked even with `language='English'`, so it needs a logit mask or transliteration step. Shares the GPU with whisper and the decision model | Medium on accent robustness. Low on fitness until the script failure is solved |
| 3 | Sarvam `saaras:v4` REST, `mode=translit` + `keyterms` | Hosted (Azure India) | Vendor-run Svarah WER 6.11 (AssemblyAI 7.45, Scribe v2 8.69, Nova-3 9.61). Independent: Sarvam models lowest WER on 13/15 Indic languages, no Indian-English split. CalQuity review: still alters names | RTT measured 45-50 ms. Processing unmeasured (no API key). Estimate 250-450 ms REST. Only third-party figure: pipecat P99 1.17 s for the legacy WebSocket | `keyterms`, max 50 terms x 64 chars, REST/batch and v4 only. WebSocket has only a free-text `prompt`. Latin-script keyterms with translit output untested | Low: multipart POST over a persistent HTTP/2 connection | Low-medium. Fallback only: audio leaves the machine, latency unverified |
| 4 | Speaker-adapted small.en (LoRA r=32 or encoder/full fine-tune, merged, converted to ggml) | Local | No direct study. Honest prior 5-35% relative WER, unknown: Zhao et al. is 3.5-15% against the un-quantised baseline; Shor 2019 35% on an RNN-T; one two-day-old anonymous HF card 55% from 1.1 h | Unchanged: 133-155 ms | Prompt kept. Whether prompt-following survives fine-tuning is an open risk; train with sampled bias prompts | High: 20-30 min enrollment plus labels, training on MPS, `convert-h5-to-ggml.py` needs `vocab.json` and `added_tokens.json` copied in (transformers 5.x no longer writes them) | Low until measured. Phase 2 |
| 5 | Parakeet TDT 0.6B v2 int8 via sherpa-onnx + hotwords | Local (CPU) | None for Indian English. African-accent proxy shows parity with Whisper-large-v3 (22.54 vs 21.19; commands 32.65 vs 31.58) | Measured 78-132 ms at 4 threads; hotwords add nothing | ContextGraph hotwords. Narrow usable window on Indian names: score 1.5-3 no effect, 5 fixed the name but deleted 'and play', 8+ looped. `bpe.vocab` missing from all tarballs (pre-tokenized hotwords work). Beam-search issue #3267 open, status on 1.13.8 unknown | Medium: `pip install sherpa-onnx`, in-process | Low. Hindi words are not in its vocabulary |
| 6 | Deepgram Nova-3 `en-IN` at `api.in.deepgram.com` | Hosted (AWS Hyderabad) | Weakest of four on vendor-run Svarah (9.61) | RTT measured 44-49 ms. Pipecat TTFS median 247 ms includes 200 ms VAD hangover, so roughly 50 ms post-VAD in-region; about 100-150 ms from here (inferred) | `keyterm`, 500 tokens (about 100 terms). Not confirmed on the India endpoint | Low | Medium on latency. Low for Hinglish: nova-3 has no romanized mode, Hindi words survive only as keyterms |
| 7 | Moonshine Medium Streaming + keyterms | Local (CPU) | None. Its own docs: key terms "cannot help with a new accent" | Measured whole-buffer 139-260 ms, 304 ms with key terms on a 2.9 s clip. The vendor's 59 ms applies only if audio is streamed in during speech | Prefix-tree logit bonus, about 1,000 terms feasible, fixed one TTS clip at default boost | Medium: only in budget as a streaming integration | Low |

Not shortlisted, so nobody re-investigates them:

- Apple SpeechTranscriber (en-IN): ignores contextualStrings entirely (Apple engineer statement plus byte-identical local output), so it fails the biasing requirement; also worse on short commands ('type hello' -> 'Thai palo.'). Its hi-IN / mul-IN locales are not downloaded, output script unknown, and have no biasing either.
- Legacy SFSpeechRecognizer via PyObjC: same model family as DictationTranscriber, adds TCC / Info.plist friction from a bare Python host, never run.
- Other whisper: medium-q5 372 ms, large-v3-turbo 478-485 ms, distil-large-v3.5 keeps the 32-layer encoder, Oriserve Apex/Prime over budget. Oriserve Swift fits (56-58 ms, Latin output) but is clearly worse on English verbs ('type hello' -> 'Hello.', 'deafen me' -> 'Diffin me.' with the phrase in the prompt). Trelis and Shunya emit Devanagari. AI4Bharat models have no English. FUTO ACFT small.en hallucinated on sub-1 s clips.
- parakeet-mlx (32-63 ms here, no biasing at all), FluidAudio (Swift-only, post-hoc CTC rescoring; its HTTP server measured 340 ms on an M4), Granite, Canary, Kyutai.
- Hosted: ElevenLabs (default routing us-central1, 316-419 ms; India residency is Enterprise-only), AssemblyAI (does support Hindi, but emits Devanagari and is EU/US hosted), OpenAI (304-367 ms RTT plus 637 ms median TTFS), Groq (same Whisper weights, no accuracy upside), Google Chirp 3 (us/eu only), Speechmatics and Gladia (EU). Azure Speech centralindia was never evaluated.

## 3. Cheap wins that need no new model

Do now. All three are exposed by stock brew `whisper-server` 1.9.2 on `/inference`.

1. Fix the bias-prompt budget.
   - whisper.cpp keeps the tail of the prompt: `max_prompt_ctx = min(n_max_text_ctx, n_text_ctx/2)` = 224, so the last 223 tokens survive. `stt.py` orders terms on the assumption that the front survives.
   - The production prompt is 50 terms / 517 chars / 207 tokens. Nothing is cut today, but headroom is 16 tokens and any growth silently deletes the highest-priority artists.
   - `hinglish.json` supplies 60 terms against the 50-term cap, so `own_names()` and app names never reach the prompt. Contact and app names are currently unbiased.
   - Fix: count tokens with the small.en tokenizer at build time, hard-cap near 215, put highest priority last, budget per category, select per context (frontmost app: Spotify -> artists, WhatsApp/Discord -> contacts).
   - Whether the 30 Hindi function words earn their slots is untested; A/B it on the eval set.
2. `audio_ctx` on stock small.en, per request.
   - 8 TTS clips: 1500 -> 108 ms, 768 -> 68 ms, 512 -> 51 ms, 256 -> 53 ms, dynamic (`len_s*50+32`) -> 51 ms. Zero repetition loops or stalls across 120 requests; mild drift on two Hinglish clips.
   - Start at 768. Never use it on turbo-class weights (loops and 1-3 s stalls reproduced locally).
   - Value: frees 40-55 ms that pays for a second engine or a fallback call. Gate on real-voice data.
3. `response_format=verbose_json` with `no_language_probabilities=true`.
   - Per-token `probability` and segment `avg_logprob` at no cost (137 ms vs 132 ms). Without the flag it costs about +100 ms.
   - Never combine with `no_timestamps=false` ('type hello' -> 'Thank you for watching.', runs up to 2.9 s).
   - Use it for logging into the eval sidecar only. The array is per token, not per word.

Exposed, but do not use.

- Probability gate for verb snapping: unreliable. A correct ' Type' scored 0.089; a wrong ' Def' / 'en' scored 0.611 / 0.292.
- `beam_size=5`: +39 ms median, no transcript improved on 8 clips, two Hinglish clips got worse, and one 0.91 s clip took 1.3-2.3 s on five consecutive runs (`temperature_inc=0` did not fix it). `beam_size=2` and `best_of=5` were latency-safe and showed no accuracy gain.
- Example commands at the end of the prompt: no evidence they help verbs, and only 16 tokens of headroom.
- Temperature fallback: already shown to be a non-issue on the server path.
- `carry_initial_prompt`, `suppress_nst`, entropy/logprob/no-speech thresholds: exposed, not evaluated.

Not exposed by whisper-server, so not cheap.

- GBNF grammar and `grammar_penalty`: libwhisper, whisper-cli and examples/command only. Routes are a roughly 30-line patch to `examples/server/server.cpp` plus a source build, or ctypes against `/opt/homebrew/lib/libwhisper.dylib`. It forces beam search, has an unfixed partial-word truncation bug (#2496, closed by the stale bot), needs `--grammar-rule` as well as `--grammar` (#2159), and has no published accuracy numbers.
- Logit bias: not on the server. pywhispercpp's `logits_filter_callback` cannot do it (refuted): it hands Python one float by value and assignment raises TypeError. Its `suppress_regex` only suppresses.
- n-best lists: not available over HTTP. The practical substitute is cross-engine hypotheses (section 5).
- Guided-mode closed-set verb classifier (examples/command): library only.

Already in the project.

- Verb-lexicon snapping with an Indian-accent voicing map exists: `alfred_computer_use/verbs.py` (VERBS, `_VOICING` d->t b->p g->k v->f z->s, listed slips 'diap' / 'dayeb' / 'daeep' -> 'type', dictionary guard, verb-slot regex) plus `alfred_computer_use/data/phonetics.json` ('defend me' -> 'deafen me') via `stt.correct()`.
- Remaining work is rule hygiene. Accept a rule only if it is mined from at least 3 distinct TRAIN utterances, fires zero times on a negative corpus of previously-correct transcripts plus dictation text, improves DEV, and is position-constrained. Log per-rule fire counts in production.

## 4. Bench plan

### 4.1 Passive capture

- Hook `WhisperServer.transcribe()` in `alfred_computer_use/stt.py` and the Brain result. At each Silero endpoint write `<ts>.wav` (16 kHz mono, exactly the buffer sent to STT) and `<ts>.json`.
- Sidecar fields: raw transcript; text after `correct()`; bias terms sent; model id and params; frontmost app; (action, args) executed; STT ms; macOS build; mic device; optional accent-mode tag; implicit-failure flags (repeat within ~10 s, undo, cancel); per-token probabilities.
- Guards: local disk only; skip capture while a call app holds the mic; nothing is uploaded unless the user opts in to the hosted arm.
- Labeling: a weekly 5-minute review tool. Play the clip, Enter to accept or edit the text, re-run the Brain on the corrected text to propose gold (action, args), confirm.
- Label all utterances for the first two weeks, or a uniform random 30%. Also label every flagged failure. Store `stratum` = `uniform` or `failure`. Report rates on `uniform` only; `failure` is for rule mining and training.
- One-off scripted enrollment, 200-300 prompts, 20-30 min, TRAIN only: every verb in 5+ carrier phrases; minimal pairs (type/tab/tap, deafen/defend, pause/pose, Karan/Quran, word-initial p/b, t/d, k/g); top 50 entities; dictation payloads; each in every accent he uses.

### 4.2 Splits and size

- Split by time and session into TRAIN / DEV / frozen TEST. Keep some entities and phrasings TEST-only. Log every look at TEST. Rotate a fresh week into TEST monthly.
- TEST target: at least 300 uniform-stratum utterances, at least 20 per major verb, 100 entity-bearing, 60 Hinglish, 60 short commands (under 1.0 s or 3 words or fewer), plus about 100 negatives (music, side speech, noise).
- Why: n=30 gives +/-16-18 points (17/30 -> Wilson [39,73]; 14/30 -> [30,64]), so the existing result cannot rank models. Wilson half-width at a 70% rate: 8.8 (n=100), 6.3 (200), 5.2 (300), 4.0 (500). Paired McNemar at 80% power: about 85 clips to detect 15 points, 155-233 for 10 points, about 470 for 5 points.

### 4.3 Arms

All run offline on the same wavs, with the same per-clip bias terms from the sidecar truncated to each engine's cap. Every arm's text goes through the same `correct()` and the same Brain.

- A0 control: whisper-server small.en with the production prompt as logged.
- A1: A0 with the fixed prompt budget. A2: A1 with `audio_ctx=768`.
- B: Apple DictationTranscriber en-IN, `[.shortForm]`, contextualStrings up to 100, finished-buffer path, one analyzer per utterance. Run once more with `.atypicalSpeech`; it changed nothing on TTS, which says nothing about a real accent.
- C: Qwen3-ASR-0.6B-8bit via mlx-audio, `language='English'`, up to 50 `hotwords`. Report raw, and with a guard that transliterates or rejects any output containing U+0900-U+097F.
- D: Sarvam `saaras:v4` REST, `mode=translit`, up to 50 `keyterms`, persistent HTTP/2, `language_code` run three ways (hi-IN, en-IN, unknown). Needs an API key and explicit opt-in.
- Derived at no cost: rule-arbitrated A1+B, oracle(A1,B), oracle(A1,C). The oracle gap is the ceiling for any second engine.
- Optional E: Parakeet v2 int8 via sherpa-onnx, greedy and `modified_beam_search` with hotwords; count empty and 'Yeah.' outputs.

### 4.4 Metrics, in priority order

1. Action-match rate: audio -> executed (action, args) vs gold, reusing the scoring shape in `bench.py`.
2. Verb accuracy in the verb slot.
3. Entity accuracy: exact match after catalog normalization, reported separately for entities in the bias list and not in it.
4. Harmful-action rate (a wrong action executed) vs reject / no-op.
5. False-accept rate on negatives.
6. Latency p50 / p95 / max, overall and for the 2-4 s bucket.
7. Non-Latin leak rate: any clip whose final text contains U+0900-U+097F.
8. Empty or hallucinated output on clips under 1 s.
9. WER / CER and a word-onset confusion tally (t->d, p->b, k->g): diagnostics only.

Wilson CIs on every rate. Paired McNemar or paired bootstrap clustered by session for A/B.

Timing protocol: wall clock from "finished buffer handed to the engine" (the VAD end event) to final text; warm; 3 passes; whisper-server and the decision model loaded, so contention is realistic. Measure D from his line with the connection reused. Do not compare against pipecat TTFS figures without subtracting the 200 ms VAD hangover they include.

### 4.5 Pass / fail, fixed before anyone looks at TEST

Hard gates. Failing any one makes an engine ineligible as primary or parallel engine.

- Latency on 2-4 s utterances: p50 <= 200 ms and p95 <= 300 ms. A parallel engine also gets a 300 ms timeout that falls back to A.
- Script: 0 clips with Devanagari in the final text on the 60+ Hinglish clips. 0/60 still allows a true rate up to about 5% at 95% confidence.
- False-accept on negatives <= control + 2 points.
- Harmful-action rate <= control.
- Short-command slice verb accuracy >= control - 3 points. For a parallel engine this applies to the arbitrated output.
- Empty or hallucinated output on sub-1 s clips <= control.

Win conditions.

- Replace A: action-match >= control + 10 points on uniform TEST, McNemar p < 0.05, all gates pass.
- Add as a second engine: arbitrated A1+X >= A1 + 10 points action-match (p < 0.05). Or >= +15 points entity accuracy on the entity-bearing slice (p < 0.05) with no short-command loss, in which case route only that slice to X.
- Kill at the n=100 interim look: if oracle(A1,X) - A1 < 5 points the engine adds nothing; stop work on it.
- Hosted fallback D: judged on the trigger set only (A1 unparsed, or engines disagree). It must recover at least 40% of A1's failures there, the trigger must fire on no more than 20% of uniform utterances, and the call p95 must be <= 500 ms. It exceeds the 300 ms budget by design; that is acceptable only because the alternative is a wrong action or a repeat.
- A1: adopt if entity accuracy improves and nothing else regresses beyond noise. A2: adopt if action-match is within 2 points of A1 and there are no new hallucinations on sub-1 s clips.

Phase 2, once 20-30 min of TRAIN audio exists: speaker-adapted small.en. Compare LoRA r=32 on q,k,v,out,fc1,fc2 against encoder-only and full fine-tuning; sample bias prompts into the training examples; mix in 20-30% generic English; select on DEV by action-match; then apply the same gates on TEST. Re-test stock medium on real voice as a diagnostic only (372 ms is over budget), because on real Indian speech the substitution rate falls with model size.

## 5. Integration sketch for the #1 pick

Apple DictationTranscriber (en-IN) as a parallel second engine beside whisper-server.

Process model.

- One single-file Swift daemon built with `swiftc -O -parse-as-library`. No Xcode project, app bundle or Info.plist. Buffer input only, so no TCC prompt (confirmed). Mic capture stays in Python.
- Python spawns it once at startup over stdin/stdout pipes or a Unix socket, supervises it, and restarts it on exit.
- One analyzer per utterance. A long-lived analyzer with `finalize(through: nil)` hung in the probe.
- The daemon keeps one pre-built "next" analyzer: `DictationTranscriber(locale: en-IN, contentHints: [.shortForm])` -> `SpeechAnalyzer` -> `setContext` (contextualStrings[.general] = bias) -> `prepareToAnalyze`. Building with `setContext` costs a median 169 ms and up to about 456 ms, so rebuild right after each utterance and on every bias change, never on the hot path. If an utterance arrives while the next analyzer is still building, return "not ready" and let whisper answer alone.
- Per utterance: convert float32 -> Int16 (the analyzer format is 1 ch, 16 kHz, Int16, no resample), `start(inputSequence:)`, yield `AnalyzerInput(buffer:)`, finish, `finalizeAndFinishThroughEndOfInput()`. In non-progressive mode results arrive flagged `isFinal=false`; take the last result after finish.

Warm-up.

- At start, check that `DictationTranscriber.installedLocales` contains en-IN. `AssetInventory.status` returns 'supported' even for installed locales, so do not use it. If en-IN is missing, report the error and do not auto-download; an asset download is a system change that needs consent.
- `reservedLocales` is empty on this Mac: en-IN is present only because the OS's own dictation installed it, so it could disappear. Check on every start.
- Run one throwaway transcription to load the model. Process start to first transcript measured 164-341 ms with assets resident. True cold start after a reboot is unmeasured.

Wire protocol.

```
request  := u32le header_len | header_json | u32le n_samples | float32le[n_samples]   (16 kHz mono)
header   := {"id": 17, "op": "transcribe"}
          | {"id": 18, "op": "set_bias", "terms": [... max 100, 1-2 words each]}      (n_samples = 0)
          | {"id": 19, "op": "ping"}                                                  (n_samples = 0)
response := one JSON line: {"id": 17, "text": "...", "ms": 81, "err": null}
```

Python API shape.

```python
@dataclass(frozen=True)
class SttResult:
    text: str
    ms: float
    engine: str  # "whisper" | "apple"

class AppleDictation:
    def start(self) -> None: ...            # spawn sidecar, wait for ready; raise if en-IN is not installed
    def set_bias(self, terms: Sequence[str]) -> None: ...   # max 100 phrases; rebuilds the analyzer off the hot path
    def transcribe(self, pcm: np.ndarray, timeout_s: float = 0.30) -> SttResult | None: ...
                                            # None on timeout, error, or analyzer not ready
    def close(self) -> None: ...
```

Call path: submit `whisper.transcribe(pcm, prompt)` and `apple.transcribe(pcm)` concurrently at VAD end; wait for whisper, wait for Apple up to its timeout; run `correct()` on both; arbitrate. Wall time is the maximum of the two, not the sum (whisper 133-155 ms; Apple 77-92 ms median, p95 207).

Bias set: up to 100 phrases of 1-2 words, chosen per context from the 1016-entity catalog with the same selector that feeds the whisper prompt. `setContext` replaces the whole context.

Arbitration, initial rule. Tune on DEV only.

1. Normalize both: lowercase, strip punctuation. Apple capitalizes oddly ('Gaana', 'Ye') and emits no punctuation with `transcriptionOptions []`.
2. Equal -> done.
3. Apple empty, or the utterance is under about 1.0 s or 3 words or fewer -> whisper. This covers Apple's measured short-clip weakness.
4. Otherwise score each hypothesis: a known verb in the verb slot, plus each span that resolves to a catalog entity. Higher score wins; tie -> whisper.
5. Untested option: hand both hypotheses to the Brain as a 2-best list. This is the only n-best path available without patching whisper.cpp.

Fallback ladder.

- Sidecar dead, timed out, not ready, or locale missing -> whisper only, which is today's behaviour.
- Both engines fail to parse and the user has opted in -> Sarvam `saaras:v4` REST, `mode=translit`, up to 50 keyterms, persistent HTTP/2 (a fresh TCP + TLS handshake costs about 107 ms).

Operations.

- The model is OS-managed and unversioned. Log the macOS build in every eval sidecar and re-run the regression set after OS updates.
- Tests: protocol unit tests against a fake sidecar; arbitration as a pure function with table-driven tests; integration tests on recorded wavs only. No live mic or GUI tests while the user is active.
- Later option: stream the VAD's 100 ms chunks into the analyzer during speech. Post-speech latency measured 29-43 ms median with a corrected probe (first volatile partial 530-900 ms after speech start). It could replace the base.en partials tier.

## 6. What remains unverified

Accuracy.

- Everything about accuracy on the real user. All local clips were Apple TTS or `say`; Apple TTS into Apple ASR is a best case.
- No published Indian-accent benchmark exists for Apple Dictation, SpeechTranscriber, Parakeet or Moonshine. The `theshivam7/indian-asr-bench` repo that reportedly covered Parakeet on Svarah returns 404 everywhere.
- "Bigger whisper did not help" is statistically unproven: the Wilson CIs of 17/30, 14/30 and 12/30 overlap almost completely (best-case McNemar p = 0.0625-0.25), and it was TTS. On real Indian-accented read speech medium beats small (13.2 vs 17.6) and substitution rate falls with size (12.92 / 9.99 / 8.32%), while large-v3 hallucinates (insertion 9.62%). That study used multilingual checkpoints and read sentences, not commands.

Apple.

- Latency when run in parallel with whisper on Metal and the decision model. It was measured under unrelated CPU load only.
- True cold start after reboot.
- Short-utterance fragility has no known mitigation: 400 ms of silence padding fixed one clip and broke another, and the `.phrase` preset changed nothing.
- The custom language model changed zero transcripts in two attempts, the second with valid X-SAMPA symbols.
- Whether a long-lived analyzer works with explicit `finalize(through:)` timestamps (not tried).
- Whether the en-IN asset stays installed when nothing reserves it.
- The hi-IN and mul-IN SpeechTranscriber models: not downloaded, output script unknown. The claim that the 15 Indian locales are new in macOS 27 rests on a citation that did not support it.
- Silent model changes with OS updates.

Qwen3-ASR.

- Devanagari leaked with `language='English'` forced (mixed on 0.6B with 50 hotwords, full on 1.7B). Whether mlx-audio exposes a logit hook for a Devanagari mask was not checked.
- GPU contention with whisper and the decision model.
- Hotword efficacy rests on 5 TTS clips. 'deafen me' stayed 'Defend me' on the 0.6B.

Sarvam and other hosted.

- Real REST latency with keyterms + translit on a 3B decoder; no API key was available. Vendor figures are time-to-first-token on the WebSocket path, which has no keyterms.
- Whether Latin-script keyterms bias translit output. The vendor's validation used native-script entity lists.
- Whether `mode` works on v4: the API reference says v3 only; the model page, keyterms guide and pipecat's config say v4 supports it.
- How translit renders English-dominant commands such as 'type hello', which `language_code` suits them, and whether romanized name spellings match the catalog.
- Svarah numbers are vendor-run and not replicated.
- Deepgram: `keyterm` on the India endpoint is not confirmed and no GA statement was found. Legacy `nova` has an `hi-Latn` model that nobody tested.
- Azure Speech centralindia (38-93 ms RTT from here) was never evaluated. Pipecat measures its TTFS at 1016 ms median.
- Groq returned `x-groq-region: bom` for unauthenticated STT POSTs; where inference runs with a valid key is unknown. It does not matter: same Whisper weights.

Whisper-side and adaptation.

- `audio_ctx` reduction on stock small.en with a real voice.
- The cause of the `beam_size=5` stall.
- Whether prompt-following survives fine-tuning, and the real adaptation gain for this speaker.
- faster-whisper small.en + `hotwords` on CPU was never measured; it was dismissed on a turbo figure.
- sherpa-onnx #3267 (about 20% empty or hallucinated output under beam search) was filed before a related fix and never re-tested. Zero failures in 72 local decodes on 1.13.8, but only 3 distinct clips.

Sourcing.

- WebSearch ran out in every lane, so blog-only and forum-only coverage is thin.
- Could not be verified: the NVIDIA Riva quote on WER, AssemblyAI Slam-1 deprecation, "press coverage" that Oriserve Apex targets Indian-accented English.

Housekeeping.

- Probe code, clips and venvs live in the session scratchpad under `/private/tmp/claude-501/-Users-rits-development/b50c51a6-bb05-4cc8-bc63-a9ed3479712c/scratchpad/` (`applestt/`, `skeptic_apple/`, `pk/`, `v/`, `sk/`). That location is not durable. Copy `applestt/probe.swift` and `skeptic_apple/probe2.swift` into the repo before it disappears if the sidecar is going ahead.
- About 8 GB of model weights were downloaded to `~/.cache/huggingface/hub` (Qwen3-ASR 0.6B and 1.7B 8-bit, Parakeet v2 and v3), and `pk/` holds about 1.6 GB including a redundant 464 MB tarball (`pk_v2_int8.tar.bz2`). The destructive-command hook blocked deletion, so these need removing by hand.
- Nothing in `/Users/rits/development/jev-voice` was modified by any lane, and the running whisper-servers on 8178/8179 were not touched.

## Trying the Apple engine

The #1 pick is now in the repo as an opt-in: `alfred_computer_use/native/apple_stt.swift` (the sidecar), `alfred_computer_use/apple_stt.py` (builds it, keeps it warm, falls back), `scripts/bench_stt.py` (the A/B). It takes the finished sentence only; drafts stay on whisper base.en. It replaces whisper for that pass rather than running beside it, so there is no arbitration yet: that waits for the bench to say whether Apple wins on the real voice. No compiler, en-IN missing, a hang past the timeout, a crash, junk on the pipe: that utterance goes to whisper, and three failures in a row bench Apple for five minutes.

```sh
cd /Users/rits/development/jev-voice

# 1. Can this Mac run it, and does the sidecar build? (about 6 s once, then cached
#    in ~/.cache/jev-voice/bin/apple-stt and rebuilt only when the .swift changes)
.venv/bin/python -c "from alfred_computer_use import apple_stt; print(apple_stt.available(), apple_stt.build())"

# 2. Its tests. The last one builds the real sidecar and synthesises "open spotify"
#    into a file with `say -o`; nothing is played.
.venv/bin/python -m pytest tests/test_apple_stt.py -q

# 3. Use it for one session (or put STT_ENGINE=apple in .env). The banner still says
#    whisper; the line starting "Sentences go to Apple's en-IN recogniser" is the proof.
STT_ENGINE=apple jev

# 4. After some real use the recorder has clips in ~/.cache/jev-voice/utterances:
.venv/bin/python scripts/bench_stt.py --limit 20                  # side by side, and how often they disagree
.venv/bin/python scripts/bench_stt.py --limit 20 --label          # type what you actually said; both get scored
.venv/bin/python scripts/bench_stt.py --limit 20 --label --play   # the same, playing each clip OUT LOUD first
.venv/bin/python scripts/bench_stt.py --dir ~/some/wavs --whisper-port 8199   # any folder; a running assistant's whisper is left alone
```

In `--label`, Enter accepts the first engine's transcript (`--engines apple,whisper` swaps which), `s` skips, `q` stops; labels land in `labels.jsonl` beside the clips as they are typed. Read before pressing Enter: accepting a wrong transcript scores that engine as right.

Knobs: `STT_ENGINE=apple`, `APPLE_STT_LOCALE` (en-IN), `APPLE_STT_TIMEOUT_MS` (1500). The sidecar's stderr goes to `~/.cache/jev-voice/apple-stt.log`. If it reports the en-IN assets missing, `~/.cache/jev-voice/bin/apple-stt en-IN --install < /dev/null` downloads them; that is never done automatically.

Measured with the built engine (macOS 27.0, `say -v Rishi` / `-v Aman` into files, the project's 56 bias terms as context for both, whisper small.en on a spare port). Still a synthetic voice, which is the whole reason for step 4.

| said | Apple, Rishi | Apple, Aman | whisper small.en, Rishi | whisper small.en, Aman |
|---|---|---|---|---|
| type hello how are you | Type hello how are you (87 ms) | Type hello how are you (71) | Type hello how are you. (174) | Type hello how are you. (181) |
| open spotify | Open Spotify (72) | Open Spotify (65) | Open Spotify. (161) | Open Spotify. (172) |
| play karan aujla | Play Karan Aujla (64) | Play Karan Aujla (65) | **Click on Aujla** (177) | **Flay, karan, aujla.** (216) |
| send it | Send it (65) | Send it (46) | send it (197) | Send it. (160) |
| pause | Pause (55) | **Pass** (55) | pause (170) | **Fosk.** (168) |
| open my chat with maa on whatsapp | Open my chat with maa on WhatsApp (100) | Open my chat with MA on WhatsApp (82) | open my chat with **ma** on whatsapp. (182) | open my chat with **ma** on whatsapp. (184) |
| play the second video | Play The second video (75) | Play The second video (67) | Play the second video. (192) | Play the second video. (193) |
| deafen me | **Defend me** (79; `correct()` repairs it) | **Definitely** (84) | Defen me. (184; repaired) | **Defenme.** (274) |

Median 71 ms against 182 ms. Over 400 back-to-back utterances the sidecar held 21 MB and an 87-97 ms median with no answer changing, and answered in 64-88 ms after 90 s idle.

Two things the sketch in section 5 got right and the probe's "persist" branch did not: an endless input stream with `finalize(through: nil)` never returns on this machine, and one analyzer fed sequence after sequence hears the session as a single dictation (" tight", " def me", "APDhillon"). So the sidecar builds a fresh analyzer per utterance, ahead of time. "deafen me" is not among the 56 bias terms; Apple takes 100, so 44 slots are spare for exactly that kind of phrase.
