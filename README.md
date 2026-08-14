========================================================================
   WHISPERFLOW — AUTOMATED STT WER BENCHMARK REPORT
   Generated     : 06 July 2026, 02:03 PM
   Test sentences: 1000
   Total words   : 14,458
   Avg words/sent: 14.5
   TTS engine    : espeak-ng v1.51 (offline, 16 kHz mono WAV)
   STT models    : ElevenLabs Scribe v2 | Deepgram Nova-2 | Smallest.ai Lightning
   WER library   : jiwer 4.0
========================================================================

────────────────────────────────────────────────────────────────────────
  PIPELINE EXPLANATION
────────────────────────────────────────────────────────────────────────
  Stage 1 — TTS (Text to Speech)
    Each sentence converted to 16 kHz mono WAV via espeak-ng offline.
    Files cached in ./audio_cache — reruns skip already generated files.
    espeak-ng chosen over gTTS: no internet, no rate limits, 1000 files instant.

  Stage 2 — STT (Speech to Text) — 3 models
    The EXACT same WAV file sent to all 3 APIs for a fair comparison.
    a) ElevenLabs Scribe v2 — POST https://api.elevenlabs.io/v1/speech-to-text
    b) Deepgram Nova-2      — POST https://api.deepgram.com/v1/listen
    c) Smallest.ai Lightning— POST https://api.smallest.ai/waves/v1/pulse/get_text
    Sequential processing with 0.4s delay to respect rate limits.
    Auto-retries 3x with 1s/2s/4s exponential backoff on failure.

  Stage 3 — WER Calculation
    All transcripts lowercased before comparison.
    jiwer.process_words() aligns reference and hypothesis using edit distance.
    WER (%) = (Substitutions + Deletions + Insertions) / Total Reference Words x 100
    Word Accuracy (%) = 100 - WER

  Sentence categories (1000 total):
    Technical / CS        : 150   Numbers / Stats   : 80
    Medical / Biology     : 100   Legal / Formal    : 80
    Financial / Economics : 80    Science / Physics : 100
    Proper Nouns / History: 100   Long / Complex    : 150
    Everyday / Complex    : 160

  WER benchmarks: <5% Excellent | 5-10% Good | 10-20% Fair | >20% Poor

────────────────────────────────────────────────────────────────────────
  MODEL : ElevenLabs Scribe v2
────────────────────────────────────────────────────────────────────────
  WER              : 11.56%
  Word Accuracy    : 88.44%
  Sentences Tested : 1000
  Total Words      : 14,458
  Total Errors     : 1,671
    Substitutions  : 802
    Deletions      : 866
    Insertions     : 3
  API Call Errors  : 0

────────────────────────────────────────────────────────────────────────
  MODEL : Deepgram Nova-2
────────────────────────────────────────────────────────────────────────
  WER              : 1.47%
  Word Accuracy    : 98.53%
  Sentences Tested : 1000
  Total Words      : 14,458
  Total Errors     : 213
    Substitutions  : 147
    Deletions      : 28
    Insertions     : 38
  API Call Errors  : 0

────────────────────────────────────────────────────────────────────────
  MODEL : Smallest.ai Lightning
────────────────────────────────────────────────────────────────────────
  WER              : 13.77%
  Word Accuracy    : 86.23%
  Sentences Tested : 625
  Total Words      : 9,029
  Total Errors     : 1,243
    Substitutions  : 996
    Deletions      : 228
    Insertions     : 19
  API Call Errors  : 375

────────────────────────────────────────────────────────────────────────
  FINAL RANKING
────────────────────────────────────────────────────────────────────────
  Rank           Model                          WER   Accuracy   Errors
  1st  WINNER    Deepgram Nova-2              1.47%     98.53%      213
  2nd            ElevenLabs Scribe v2        11.56%     88.44%    1,671
  3rd            Smallest.ai Lightning       13.77%     86.23%    1,243

  WINNER : Deepgram Nova-2
    WER      : 1.47%
    Accuracy : 98.53%

========================================================================
  END OF REPORT
========================================================================
