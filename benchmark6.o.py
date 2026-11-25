"""
WhisperFlow — Automated STT WER Benchmark
==========================================
Compares ElevenLabs Scribe v2, Deepgram Nova-2, Smallest.ai Lightning
on 1000 complex sentences using a fully automated pipeline.

HOW TO GET API KEYS:
  ElevenLabs : https://elevenlabs.io/app/settings/api-keys
               (Free tier: 30 min STT/month — enough for this test)
  Deepgram   : https://console.deepgram.com/
               (Free $200 credit on signup — no card needed)
  Smallest.ai: https://waves.smallest.ai/
               (Free tier available — signup and get key from dashboard)

USAGE:
  python benchmark.py \
      --elevenlabs-key sk_xxx \
      --deepgram-key   dg_xxx \
      --smallest-key   sm_xxx

  Or set env vars instead of flags:
      export ELEVENLABS_API_KEY=sk_xxx
      export DEEPGRAM_API_KEY=dg_xxx
      export SMALLEST_API_KEY=sm_xxx
      python benchmark.py

  Skip any model you don't have a key for:
      python benchmark.py --elevenlabs-key sk_xxx --skip-deepgram --skip-smallest

PIPELINE:
  Stage 1 — TTS   : espeak-ng converts each sentence to 16kHz mono WAV (offline)
  Stage 2 — STT   : Same WAV sent to each API model (REST HTTP POST)
  Stage 3 — WER   : jiwer computes substitutions + deletions + insertions
  Stage 4 — Report: Full ranked table saved to wer_report.txt

INSTALL DEPS:
  pip install elevenlabs deepgram-sdk jiwer requests
  sudo apt install espeak-ng ffmpeg   (Linux)
  brew install espeak ffmpeg          (macOS)
  Windows: download espeak-ng from github.com/espeak-ng/espeak-ng/releases
"""

import os, sys, time, argparse, subprocess, requests, io, traceback
from pathlib import Path
from datetime import datetime
from elevenlabs.client import ElevenLabs
from deepgram import DeepgramClient
from jiwer import process_words

sys.path.insert(0, str(Path(__file__).parent))
from sentences import SENTENCES

# ── colours ──────────────────────────────────────────────
R="\033[31m"; G="\033[32m"; Y="\033[33m"
B="\033[34m"; M="\033[35m"; C="\033[36m"
W="\033[0m";  BD="\033[1m"

def log(colour, tag, msg):
    print(f"{BD}{colour}[{tag}]{W} {time.strftime('%H:%M:%S')}  {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — TTS: sentence → 16kHz mono WAV using espeak-ng (offline)
# WHY espeak-ng:  gTTS needs internet + gets rate limited at 1000 calls.
#                 espeak-ng is local, instant, free, no rate limits.
# ══════════════════════════════════════════════════════════════════════

def sentence_to_wav(sentence: str, wav_path: Path) -> bool:
    """
    Convert sentence to WAV using gTTS + ffmpeg.
    Install: pip install gtts
    ffmpeg Windows: https://ffmpeg.org/download.html -> add to PATH
    """
    try:
        from gtts import gTTS
        mp3_path = wav_path.with_suffix(".mp3")
        tts = gTTS(text=sentence, lang="en", slow=False)
        tts.save(str(mp3_path))
        ffm = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path),
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(wav_path)],
            capture_output=True, timeout=30,
        )
        try:
            mp3_path.unlink()
        except Exception:
            pass
        return ffm.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0
    except Exception as e:
        log(R, "TTS", f"Error: {e}")
        return False

def generate_all_audio(sentences: list, audio_dir: Path) -> list:
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths, generated, cached, failed = [], 0, 0, 0
    log(B, "TTS", f"Generating audio for {len(sentences)} sentences (gTTS + ffmpeg) ...")
    for i, sent in enumerate(sentences):
        wav = audio_dir / f"s{i:05d}.wav"
        if wav.exists() and wav.stat().st_size > 1000:
            paths.append(wav); cached += 1
        elif sentence_to_wav(sent, wav):
            paths.append(wav); generated += 1
        else:
            paths.append(None); failed += 1
        if (i + 1) % 100 == 0:
            log(G, "TTS", f"  {i+1}/{len(sentences)} — new:{generated} cached:{cached} failed:{failed}")
    log(G, "TTS", f"Done — {len(sentences)-failed}/{len(sentences)} files ready")
    return paths

# ══════════════════════════════════════════════════════════════════════
# STAGE 2a — ElevenLabs Scribe v2
# API docs: https://elevenlabs.io/docs/api-reference/speech-to-text
# Endpoint: POST https://api.elevenlabs.io/v1/speech-to-text
# Auth    : xi-api-key header (handled by SDK)
# ══════════════════════════════════════════════════════════════════════

def transcribe_elevenlabs(wav_path: Path, client: ElevenLabs) -> str:
    audio_file = io.BytesIO(wav_path.read_bytes())
    audio_file.name = "audio.wav"
    result = client.speech_to_text.convert(
        model_id="scribe_v2",
        file=audio_file,
        language_code=None,         # auto-detect language
        tag_audio_events=False,
        timestamps_granularity=None,
    )
    # result.text is the transcript string
    return (result.text or "").strip().lower()

# ══════════════════════════════════════════════════════════════════════
# STAGE 2b — Deepgram Nova-2
# API docs: https://developers.deepgram.com/reference/pre-recorded
# Endpoint: POST https://api.deepgram.com/v1/listen
# Auth    : Token header (handled by SDK)
# ══════════════════════════════════════════════════════════════════════

def transcribe_deepgram(wav_path: Path, client: DeepgramClient) -> str:
    audio_bytes = wav_path.read_bytes()
    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-2",
        language="en",
        punctuate=False,    # no punctuation — pure words for fair WER comparison
        smart_format=False, # no smart formatting
        numerals=False,     # keep numbers as words matching our reference
    )
    try:
        # response.results.channels[0].alternatives[0].transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        return (transcript or "").strip().lower()
    except (AttributeError, IndexError, TypeError):
        return ""

# ══════════════════════════════════════════════════════════════════════
# STAGE 2c — Smallest.ai Lightning
# API docs: https://waves.smallest.ai/docs
# Endpoint: POST https://api.smallest.ai/waves/v1/pulse/get_text
# Auth    : Bearer token in Authorization header
# ══════════════════════════════════════════════════════════════════════

def transcribe_smallest(wav_path: Path, api_key: str) -> str:
    # Correct endpoint from smallest.ai official docs:
    # https://docs.smallest.ai/waves/documentation/speech-to-text-pulse/overview
    # Sends raw audio bytes in body with Content-Type: audio/wav
    url = "https://api.smallest.ai/waves/v1/pulse/get_text"
    audio_data = wav_path.read_bytes()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "audio/wav",
    }
    params = {"language": "en"}
    resp = requests.post(url, data=audio_data, headers=headers,
                         params=params, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        # response field is "transcription" per official docs
        text = (
            data.get("transcription") or
            data.get("text") or
            data.get("transcript") or ""
        )
        return text.strip().lower()
    else:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")

# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — Batch runner with 3x retry + exponential backoff
# ══════════════════════════════════════════════════════════════════════

def run_model(model_name, transcribe_fn, audio_paths, sentences, delay=0.4):
    log(M, model_name, f"Starting — {sum(1 for p in audio_paths if p)} files ...")
    refs, hyps, errors, skipped = [], [], 0, 0

    for i, (wav, ref) in enumerate(zip(audio_paths, sentences)):
        if wav is None:
            skipped += 1
            continue

        hyp = None
        for attempt in range(3):
            try:
                hyp = transcribe_fn(wav)
                break
            except Exception as e:
                wait = 2 ** attempt
                if attempt < 2:
                    time.sleep(wait)
                else:
                    log(R, model_name, f"  [{i+1}] Failed after 3 attempts: {e}")
                    errors += 1

        if hyp is not None:
            refs.append(ref.lower())
            hyps.append(hyp)

        if (i + 1) % 100 == 0:
            log(G, model_name, f"  {i+1}/{len(audio_paths)} done — errors: {errors}")

        time.sleep(delay)

    log(G, model_name, f"Complete — {len(refs)} transcribed, {errors} errors, {skipped} skipped")
    return {"name": model_name, "refs": refs, "hyps": hyps,
            "errors": errors, "skipped": skipped}

# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — WER calculation
# WER = (S + D + I) / total_reference_words  (via jiwer)
# ══════════════════════════════════════════════════════════════════════

def calc_stats(result):
    refs = result["refs"]
    hyps = result["hyps"]
    if not refs:
        return {"wer": 100.0, "accuracy": 0.0, "total_words": 0,
                "total_errors": 0, "S": 0, "D": 0, "I": 0,
                "sentences": 0, "api_errors": result["errors"]}
    out   = process_words(reference=refs, hypothesis=hyps)
    S, D, I = out.substitutions, out.deletions, out.insertions
    total_words   = sum(len(r.split()) for r in refs)
    total_errors  = S + D + I
    wer_val       = out.wer
    return {
        "wer":          wer_val * 100,
        "accuracy":     (1 - wer_val) * 100,
        "total_words":  total_words,
        "total_errors": total_errors,
        "S": S, "D": D, "I": I,
        "sentences":    len(refs),
        "api_errors":   result["errors"],
    }

# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — Report: print to terminal + save to wer_report.txt
# ══════════════════════════════════════════════════════════════════════

def print_and_save_report(all_results, sentences):
    now        = datetime.now().strftime("%d %B %Y, %I:%M %p")
    total_words= sum(len(s.split()) for s in sentences)
    stats_list = [(r, calc_stats(r)) for r in all_results]
    ranked     = sorted(stats_list, key=lambda x: x[1]["wer"])
    medals     = ["1st  WINNER", "2nd", "3rd"]

    # ── terminal ─────────────────────────────────────────────────────
    print(); print("=" * 70)
    print("   WHISPERFLOW — AUTOMATED STT WER BENCHMARK REPORT")
    print(f"   Generated : {now}")
    print(f"   Sentences : {len(sentences)}  |  Words : {total_words}")
    print(f"   TTS       : espeak-ng (offline, 16 kHz mono WAV)")
    print(f"   WER tool  : jiwer 4.0")
    print("=" * 70)

    for r, s in stats_list:
        print(f"\n  {BD}MODEL : {r['name']}{W}")
        print(f"  {'─'*52}")
        print(f"  WER              : {BD}{s['wer']:.2f}%{W}")
        print(f"  Word Accuracy    : {BD}{s['accuracy']:.2f}%{W}")
        print(f"  Sentences Tested : {s['sentences']}")
        print(f"  Total Words      : {s['total_words']:,}")
        print(f"  Total Errors     : {s['total_errors']:,}")
        print(f"    Substitutions  : {s['S']:,}")
        print(f"    Deletions      : {s['D']:,}")
        print(f"    Insertions     : {s['I']:,}")
        print(f"  API Call Errors  : {s['api_errors']}")

    print(); print("─" * 70)
    print("  FINAL RANKING")
    print("─" * 70)
    print(f"  {'Rank':<14} {'Model':<26} {'WER':>7}  {'Accuracy':>9}  {'Errors':>7}")
    print(f"  {'─'*14} {'─'*26} {'─'*7}  {'─'*9}  {'─'*7}")
    for rank, (r, s) in enumerate(ranked):
        medal = medals[rank] if rank < len(medals) else f"{rank+1}th"
        print(f"  {medal:<14} {r['name']:<26} {s['wer']:>6.2f}%  {s['accuracy']:>8.2f}%  {s['total_errors']:>7,}")

    wr, ws = ranked[0]
    print(f"\n  {G}{BD}WINNER : {wr['name']}  —  WER {ws['wer']:.2f}%  |  Accuracy {ws['accuracy']:.2f}%{W}")
    print("=" * 70)

    # ── text file ─────────────────────────────────────────────────────
    report_path = Path(__file__).parent / "wer_report.txt"
    lines = [
        "=" * 72,
        "   WHISPERFLOW — AUTOMATED STT WER BENCHMARK REPORT",
        f"   Generated     : {now}",
        f"   Test sentences: {len(sentences)}",
        f"   Total words   : {total_words:,}",
        f"   Avg words/sent: {total_words/len(sentences):.1f}",
        "   TTS engine    : espeak-ng v1.51 (offline, 16 kHz mono WAV)",
        "   STT models    : ElevenLabs Scribe v2 | Deepgram Nova-2 | Smallest.ai Lightning",
        "   WER library   : jiwer 4.0",
        "=" * 72,
        "",
        "─" * 72,
        "  PIPELINE EXPLANATION",
        "─" * 72,
        "  Stage 1 — TTS (Text to Speech)",
        "    Each sentence converted to 16 kHz mono WAV via espeak-ng offline.",
        "    Files cached in ./audio_cache — reruns skip already generated files.",
        "    espeak-ng chosen over gTTS: no internet, no rate limits, 1000 files instant.",
        "",
        "  Stage 2 — STT (Speech to Text) — 3 models",
        "    The EXACT same WAV file sent to all 3 APIs for a fair comparison.",
        "    a) ElevenLabs Scribe v2 — POST https://api.elevenlabs.io/v1/speech-to-text",
        "    b) Deepgram Nova-2      — POST https://api.deepgram.com/v1/listen",
        "    c) Smallest.ai Lightning— POST https://api.smallest.ai/waves/v1/pulse/get_text",
        "    Sequential processing with 0.4s delay to respect rate limits.",
        "    Auto-retries 3x with 1s/2s/4s exponential backoff on failure.",
        "",
        "  Stage 3 — WER Calculation",
        "    All transcripts lowercased before comparison.",
        "    jiwer.process_words() aligns reference and hypothesis using edit distance.",
        "    WER (%) = (Substitutions + Deletions + Insertions) / Total Reference Words x 100",
        "    Word Accuracy (%) = 100 - WER",
        "",
        "  Sentence categories (1000 total):",
        "    Technical / CS        : 150   Numbers / Stats   : 80",
        "    Medical / Biology     : 100   Legal / Formal    : 80",
        "    Financial / Economics : 80    Science / Physics : 100",
        "    Proper Nouns / History: 100   Long / Complex    : 150",
        "    Everyday / Complex    : 160",
        "",
        "  WER benchmarks: <5% Excellent | 5-10% Good | 10-20% Fair | >20% Poor",
        "",
    ]

    for r, s in stats_list:
        lines += [
            "─" * 72,
            f"  MODEL : {r['name']}",
            "─" * 72,
            f"  WER              : {s['wer']:.2f}%",
            f"  Word Accuracy    : {s['accuracy']:.2f}%",
            f"  Sentences Tested : {s['sentences']}",
            f"  Total Words      : {s['total_words']:,}",
            f"  Total Errors     : {s['total_errors']:,}",
            f"    Substitutions  : {s['S']:,}",
            f"    Deletions      : {s['D']:,}",
            f"    Insertions     : {s['I']:,}",
            f"  API Call Errors  : {s['api_errors']}",
            "",
        ]

    lines += [
        "─" * 72,
        "  FINAL RANKING",
        "─" * 72,
        f"  {'Rank':<14} {'Model':<26} {'WER':>7}  {'Accuracy':>9}  {'Errors':>7}",
    ]
    for rank, (r, s) in enumerate(ranked):
        medal = medals[rank] if rank < len(medals) else f"{rank+1}th"
        lines.append(
            f"  {medal:<14} {r['name']:<26} {s['wer']:>6.2f}%  {s['accuracy']:>8.2f}%  {s['total_errors']:>7,}"
        )

    lines += [
        "",
        f"  WINNER : {ranked[0][0]['name']}",
        f"    WER      : {ranked[0][1]['wer']:.2f}%",
        f"    Accuracy : {ranked[0][1]['accuracy']:.2f}%",
        "",
        "=" * 72,
        "  END OF REPORT",
        "=" * 72,
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(G, "DONE", f"Report saved → {report_path.resolve()}")

# ══════════════════════════════════════════════════════════════════════
# VALIDATION — checks API keys work before running 1000 sentences
# ══════════════════════════════════════════════════════════════════════

def validate_key_elevenlabs(api_key: str, test_wav: Path) -> bool:
    try:
        client = ElevenLabs(api_key=api_key)
        audio_file = io.BytesIO(test_wav.read_bytes())
        audio_file.name = "audio.wav"
        result = client.speech_to_text.convert(
            model_id="scribe_v2", file=audio_file, language_code=None,
            tag_audio_events=False, timestamps_granularity=None,
        )
        text = (result.text or "").strip()
        log(G, "VALIDATE", f"ElevenLabs OK — got: '{text[:60]}'")
        return True
    except Exception as e:
        log(R, "VALIDATE", f"ElevenLabs FAILED: {e}")
        return False

def validate_key_deepgram(api_key: str, test_wav: Path) -> bool:
    try:
        client = DeepgramClient(api_key=api_key)
        audio_bytes = test_wav.read_bytes()
        response = client.listen.v1.media.transcribe_file(
            request=audio_bytes, model="nova-2", language="en",
            punctuate=False, smart_format=False, numerals=False,
        )
        text = response.results.channels[0].alternatives[0].transcript or ""
        log(G, "VALIDATE", f"Deepgram OK — got: '{text[:60]}'")
        return True
    except Exception as e:
        log(R, "VALIDATE", f"Deepgram FAILED: {e}")
        return False

def validate_key_smallest(api_key: str, test_wav: Path) -> bool:
    try:
        url = "https://api.smallest.ai/waves/v1/pulse/get_text"
        audio_data = test_wav.read_bytes()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "audio/wav",
        }
        params = {"language": "en"}
        resp = requests.post(url, data=audio_data, headers=headers,
                             params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            text = (data.get("transcription") or
                    data.get("text") or
                    data.get("transcript") or "")
            log(G, "VALIDATE", f"Smallest.ai OK — got: '{text[:60]}'")
            return True
        else:
            log(R, "VALIDATE", f"Smallest.ai FAILED: HTTP {resp.status_code} — {resp.text[:200]}")
            return False
    except Exception as e:
        log(R, "VALIDATE", f"Smallest.ai FAILED: {e}")
        return False

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Automated STT WER Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WHERE TO GET API KEYS:
  ElevenLabs : https://elevenlabs.io/app/settings/api-keys
               Free tier: 30 min STT/month (enough for this test)
  Deepgram   : https://console.deepgram.com/
               Free $200 credit on signup, no card needed
  Smallest.ai: https://waves.smallest.ai/
               Free tier available — get key from dashboard after signup

EXAMPLE:
  python benchmark.py \\
      --elevenlabs-key sk_abc123 \\
      --deepgram-key   dg_abc123 \\
      --smallest-key   sm_abc123
        """
    )
    # API keys are read from the flag or the environment variable only.
    # Never hardcode a key as a default here.
    parser.add_argument("--elevenlabs-key", default=os.getenv("ELEVENLABS_API_KEY"))
    parser.add_argument("--deepgram-key",   default=os.getenv("DEEPGRAM_API_KEY"))
    parser.add_argument("--smallest-key",   default=os.getenv("SMALLEST_API_KEY"))
    parser.add_argument("--audio-dir",      default="./audio_cache",
                        help="Where to store generated WAV files (default: ./audio_cache)")
    parser.add_argument("--limit",  type=int, default=1000,
                        help="Number of sentences to test (default: 1000)")
    parser.add_argument("--delay",  type=float, default=0.4,
                        help="Seconds between API calls (default: 0.4)")
    parser.add_argument("--skip-validate",   action="store_true",
                        help="Skip API key validation step")
    parser.add_argument("--skip-elevenlabs", action="store_true")
    parser.add_argument("--skip-deepgram",   action="store_true")
    parser.add_argument("--skip-smallest",   action="store_true")
    args = parser.parse_args()

    sentences   = SENTENCES[:args.limit]
    total_words = sum(len(s.split()) for s in sentences)
    audio_dir   = Path(args.audio_dir)

    print(f"\n{BD}{'=' * 60}{W}")
    print(f"  {BD}WhisperFlow STT Benchmark{W}")
    print(f"  Sentences : {len(sentences)}")
    print(f"  Words     : {total_words:,}")
    print(f"  Audio dir : {audio_dir.resolve()}")
    print(f"{'=' * 60}{W}\n")

    # ── Stage 1: TTS ─────────────────────────────────────────────────
    audio_paths = generate_all_audio(sentences, audio_dir)
    print()

    # ── Validation: test each key on 1 sample before full run ────────
    if not args.skip_validate:
        test_wav = next((p for p in audio_paths if p), None)
        if test_wav:
            log(B, "VALIDATE", "Testing API keys on 1 sample before full run ...")
            if args.elevenlabs_key and not args.skip_elevenlabs:
                if not validate_key_elevenlabs(args.elevenlabs_key, test_wav):
                    log(R, "VALIDATE", "ElevenLabs key failed — skipping this model")
                    args.skip_elevenlabs = True
            if args.deepgram_key and not args.skip_deepgram:
                if not validate_key_deepgram(args.deepgram_key, test_wav):
                    log(R, "VALIDATE", "Deepgram key failed — skipping this model")
                    args.skip_deepgram = True
            if args.smallest_key and not args.skip_smallest:
                if not validate_key_smallest(args.smallest_key, test_wav):
                    log(R, "VALIDATE", "Smallest.ai key failed — skipping this model")
                    args.skip_smallest = True
            print()

    # ── Stage 2: STT ─────────────────────────────────────────────────
    all_results = []

    if not args.skip_elevenlabs:
        if not args.elevenlabs_key:
            log(Y, "SKIP", "ElevenLabs: no key — use --elevenlabs-key or ELEVENLABS_API_KEY env var")
            log(Y, "SKIP", "  Get key: https://elevenlabs.io/app/settings/api-keys")
        else:
            el = ElevenLabs(api_key=args.elevenlabs_key)
            all_results.append(run_model(
                "ElevenLabs Scribe v2",
                lambda p: transcribe_elevenlabs(p, el),
                audio_paths, sentences, delay=args.delay,
            ))
            print()

    if not args.skip_deepgram:
        if not args.deepgram_key:
            log(Y, "SKIP", "Deepgram: no key — use --deepgram-key or DEEPGRAM_API_KEY env var")
            log(Y, "SKIP", "  Get key: https://console.deepgram.com/  (free $200 credit)")
        else:
            dg = DeepgramClient(api_key=args.deepgram_key)
            all_results.append(run_model(
                "Deepgram Nova-2",
                lambda p: transcribe_deepgram(p, dg),
                audio_paths, sentences, delay=args.delay,
            ))
            print()

    if not args.skip_smallest:
        if not args.smallest_key:
            log(Y, "SKIP", "Smallest.ai: no key — use --smallest-key or SMALLEST_API_KEY env var")
            log(Y, "SKIP", "  Get key: https://waves.smallest.ai/")
        else:
            sk = args.smallest_key
            all_results.append(run_model(
                "Smallest.ai Lightning",
                lambda p: transcribe_smallest(p, sk),
                audio_paths, sentences, delay=args.delay,
            ))
            print()

    if not all_results:
        log(R, "ERROR", "No models ran — provide at least one API key.")
        log(R, "ERROR", "See --help for key sources.")
        sys.exit(1)

    # ── Stage 3+4: WER + Report ───────────────────────────────────────
    print_and_save_report(all_results, sentences)

if __name__ == "__main__":
    main()
