# ASR Model Test Plan

> Synthesized from 12 reference notebooks in `lab/reference/ASR/` (2026-07-31)

---

## 1. Overview: The ASR Pipeline Architecture

Every notebook converges on a single architecture pattern:

```
Audio → [Preprocess: 16kHz mono, log-mel spectrogram]
     → [VAD: utterance segmentation]
     → [ASR Engine: adapter-wrapped]
     → [Confidence Gate + Retry]
     → [Hallucination Filter]
     → [Postprocessing: lexicon, ITN]
     → [Data Contract Adapter → dict]
```

### Engines Evaluated (12 notebooks)

| Notebook | Engine | Architecture | Korean Support | Confidence Metric | Latency Profile |
|---|---|---|---|---|---|
| 2-1, 2-3, 2-W | faster-whisper (large-v3-turbo) | Encoder-Decoder (AR) | Yes | `avg_logprob` | Fast (turbo: 4 decoder layers) |
| 2-W FT | ghost613/ft-whisper-turbo-korean | Encoder-Decoder (AR) | Yes (fine-tuned) | `avg_logprob` | Same as turbo |
| 2-S | Qwen3-ASR-0.6B | LLM-based (AR) | Yes (52 langs) | `text != ""` proxy | ~hundreds ms per utterance |
| 2-C | Cohere Transcribe | Conformer+AED (NAR) | Yes | None (needs VAD gate) | Fast, 35s auto-chunk |
| 2-M | Meta Omnilingual (CTC) | Encoder+CTC (NAR) | Yes (1,672 langs) | None | Fastest single-inference |
| 2-M | Meta Omnilingual (LLM) | Encoder+LLaMA (AR) | Yes | None | Slower (autoregressive penalty) |
| 2-V | SenseVoice-Small | CTC (NAR) | Yes | `<\|nospeech\|>` tag | ~70ms/10s audio |
| 2-L | GPT-Live (analysis only) | Full-duplex black box | Yes | N/A | Frame-level 40ms |

---

## 2. Key Metrics for ASR Model Evaluation

### 2.1 Accuracy Metrics

#### CER (Character Error Rate)
- **Formula**: `(Substitutions + Deletions + Insertions) / N`
- **Why CER over WER for Korean**: Korean spacing is inconsistent; syllable-level CER is more stable
- **Implementation**: Edit distance (Levenshtein) with `jiwer` or from-scratch numpy
- **Evaluation sets**: Clean audio + SNR 5dB (noisy) — both needed
- **Normalization policy**: Strip whitespace/punctuation before CER; match ITN conventions to ground truth

#### DER (Diarization Error Rate)
- **Formula**: `(Miss + False Alarm + Confusion) / Total Reference Speech`
- **Sub-components map to pipeline stages**:
  - Miss → VAD too conservative
  - False Alarm → VAD too sensitive
  - Confusion → Embedding/clustering failure
- **Relevant for**: Multi-speaker lecture scenarios (professor + student questions)
- **Model**: ECAPA-TDNN embeddings + cosine-distance agglomerative clustering

### 2.2 Speed Metrics

| Metric | Definition | Budget |
|---|---|---|
| **Latency (ms)** | End-of-utterance → final transcription | ≤ 500ms (ASR share of 1.5s round-trip) |
| **RTF (Real-Time Factor)** | Processing time / Audio duration | < 1.0 (streaming viability) |
| **first_partial_ms** | Stream start → first partial hypothesis | Perceived responsiveness |
| **EoU wait (ms)** | Silence threshold for utterance boundary | 300ms (aggressive) ~ 1200ms (safe) |

### 2.3 Robustness Metrics

| Metric | Signal | Threshold |
|---|---|---|
| **avg_logprob** | Whisper-native confidence | ≥ -1.0 → `confidence_ok = True` |
| **no_speech_prob** | Whisper silence detection | High → hallucination risk |
| **Speech ratio** | % of frames above RMS threshold | Near-zero + text → hallucination |
| **Token repetition** | Same token ≥ 4x | Hallucination loop |

### 2.4 Deployment Metrics

| Metric | Why It Matters |
|---|---|
| **VRAM (GB)** | T4 Colab = 16GB ceiling |
| **Model load time (s)** | Cold-start UX |
| **Batch throughput (utterances/s)** | Nightly batch QA vs real-time single-item |
| **Quantization tolerance** | int8 float16 vs float16 CER gap |

---

## 3. Processing Methods to Improve Performance

### 3.1 Preprocessing

| Method | Detail | Impact |
|---|---|---|
| **Resample → 16kHz mono** | `librosa.resample` or `soxr` | Universal ASR input requirement |
| **Log-mel spectrogram** | 80-channel, 25ms window, 10ms hop (n_fft=400, hop=160) | Shared input for all engines |
| **Hop size tradeoff** | hop=160→320 halves frames, halves ASR workload | Time resolution vs. speed |
| **Noise injection** | SNR 5dB for stress testing | Measures robustness ceiling |

### 3.2 Voice Activity Detection (VAD)

| Method | Pros | Cons |
|---|---|---|
| **Energy-based RMS** | No dependencies, fast | Fragile under noise |
| **Silero VAD v5+** | Neural, pip-installed, offline | Slightly heavier |
| **SenseVoice `<\|nospeech\|>` tag** | Built-in, no extra pass | Engine-specific |
| **silence_chunks=3 × 320ms = 960ms EoU** | Production standard | Fixed latency floor |

**Hallucination Defense**: Silence → VAD gate must block **before** ASR. All engines hallucinate from silence (Cohere, Whisper, Qwen3 all confirmed). No confidence metric catches this reliably.

### 3.3 Confidence & Retry Strategy (2-Tier)

```
1. Transcribe with greedy decoding (fast)
2. If confidence < threshold:
   → Retry with beam=5 (accurate, max 1 retry)
3. If still low confidence:
   → Return status="low_confidence" (never throw)
```

| Engine | Confidence Threshold |
|---|---|
| Whisper | `avg_logprob ≥ -1.0` |
| Qwen3-ASR | `text != ""` (proxy) |
| Cohere / Omnilingual | VAD-only gate (no confidence signal) |

### 3.4 Hallucination Detection (3 Signals)

1. **Silence paradox**: Near-zero speech ratio but text returned → filter
2. **Boilerplate patterns**: YouTube-style phrases ("구독과 좋아요"), filler loops
3. **Token repetition**: Same token ≥ 4 consecutive times → filter

### 3.5 Decoding Strategies

| Strategy | Speed | Accuracy | When to Use |
|---|---|---|---|
| **Greedy** | Fastest | Baseline | Default for real-time |
| **Beam=5** | ~3-5× slower | Better | Low-confidence retry |
| **LocalAgreement-2** | Streaming overhead | Prevents text flickering | Streaming only |
| **initial_prompt** | No overhead | Domain injection | Seed with slide context |

### 3.6 Quantization (VRAM ≤ 6GB path)

| Compute Type | VRAM | Speed | CER Impact |
|---|---|---|---|
| float16 | ~3.2GB (turbo) | Baseline | — |
| int8_float16 | ~1.6GB (turbo) | 1.1-1.3× faster | Tiny (measure, don't assume) |
| int8 | ~1.2GB | Fastest | Small |

### 3.7 Korean-Specific Postprocessing

| Method | Example | When |
|---|---|---|
| **Domain lexicon** | `{"시퀀스 다이어그램" → "시퀀스 다이어그램"}` (deterministic substitution) | Recurring errors |
| **ITN (Inverse Text Normalization)** | "삼만 원" → "30,000원" | Numeral normalization |
| **Whitespace/punctuation strip** | CER evaluation only | Before metric calculation |
| **Code-switching policy** | "payment" vs "페이먼트" → consistent rule | Mixed Korean/English |

### 3.8 Streaming Architecture

```
Audio chunks (320ms) → VAD → find_endpoint()
    → Re-transcribe growing buffer → LocalAgreement-2 commit
    → STREAM_EVENT_KEYS ⊃ CONTRACT_KEYS
```

**Key invariants**:
- `is_final=True` events must satisfy batch contract (backward compatible)
- `is_final=False` (partial) events carry stream-only fields
- Committed text never flips — LocalAgreement-2 guarantee

### 3.9 Fine-Tuning Path

**When needed**: Real phone-line audio, domain-specific terminology, accent variations

| Approach | Cost | Gain |
|---|---|---|
| Community Korean FT checkpoint | Free (download) | Clean TTS: minimal; Real noise: significant |
| Quantized FT deployment | int8 inference | VRAM tradeoff |
| `initial_prompt` injection | Zero | Domain context without fine-tuning |

### 3.10 Adapter Pattern (Universal Contract)

```python
CONTRACT_KEYS = {"utt_id", "engine", "text", "language",
                  "confidence_ok", "avg_logprob", "latency_ms"}

# Optional extension fields (don't break downstream):
# emotion, events, speaker_id
```

Every engine gets a single adapter function that maps native output → contract dict. Downstream code (logging, routing, LLM, LiveKit) never changes.

---

## 4. Evaluation Protocol for lecture-note-ai

### 4.1 Test Tracks

| Track | Audio Type | Metric | Purpose |
|---|---|---|---|
| **Clean** | Studio-quality Korean lecture | CER | Baseline accuracy |
| **Noise (SNR 5dB)** | Lecture + ambient noise | CER | Robustness |
| **Code-switched** | Korean + English technical terms | CER + manual | Domain reality |
| **Silence** | Pure silence (0dBFS) | Hallucination rate | VAD gate validation |
| **Streaming** | Simulated real-time chunks | final_latency_ms, RTF | Real-time viability |

### 4.2 Model Selection Framework

Three filters (from 2-2):

1. **Language Support**: Korean required → eliminates English-only engines (Canary-Qwen, Parakeet, Voxtral)
2. **Ecosystem Maturity**: pip-installable, T4-compatible, active community
3. **Latency Budget**: ≤500ms per utterance on T4

### 4.3 Recommended Comparison Matrix

| Engine | CER (clean) | CER (SNR 5dB) | Latency (ms) | VRAM (GB) | Confidence | Streaming |
|---|---|---|---|---|---|---|
| faster-whisper large-v3-turbo | ? | ? | ? | ~3.2 | avg_logprob | LocalAgreement-2 |
| ghost613/ft-whisper-turbo-korean | ? | ? | ? | ~3.2 | avg_logprob | LocalAgreement-2 |
| Qwen3-ASR-0.6B | ? | ? | ? | ~1.2 | text != "" | Re-transcribe |
| SenseVoice-Small | ? | ? | ? | ~1 | nospeech tag | N/A (CTC) |
| Cohere Transcribe | ? | ? | ? | ~4 | VAD-only gate | Auto-chunk |
| Meta Omnilingual CTC | ? | ? | ? | ? | VAD-only gate | Batch |

> **Fill `?` with actual measurements using the evaluation protocol above.**

---

## 5. Actionable Proposals (Priority Order)

### P0: Baseline faster-whisper + VAD Pipeline
- Implement `src/audio/asr_engine.py` with faster-whisper large-v3-turbo
- Implement `src/audio/recorder.py` with Silero VAD + EoU detection
- Run all 5 test tracks (4.1), record results in comparison matrix (4.3)
- **Rationale**: Whisper is industry standard; ecosystem mature; `avg_logprob` is the only native confidence signal

### P1: Hallucination Defense
- Implement 3-signal filter from session 2-3
- Test with silence track → zero false text tolerated
- Integrate before LLM injection (hallucinated text poisons downstream summaries)

### P2: 2-Tier Confidence Strategy
- Greedy default, beam=5 retry on low avg_logprob
- Max 1 retry per utterance → hard latency cap
- Return status dict (never throw) for LLM graceful recovery

### P3: Korean Postprocessing
- Domain lexicon for course-specific terminology
- ITN for numeral normalization
- Code-switching policy (English terms → Hangul or preserve?)

### P4: Alternative Engine Evaluation
- Run Qwen3-ASR-0.6B: smaller VRAM, built-in LID, code-switching strength
- Run SenseVoice-Small: CTC speed + emotion/event tags (bonus features)
- Swap via adapter pattern → zero pipeline code change

### P5: Korean Fine-Tuned Whisper
- Test ghost613/ft-whisper-turbo-korean on real lecture audio
- Compare CER against base turbo on clean + noisy tracks
- If gain is significant (>5% CER reduction), make it the default

### P6: Streaming Integration
- Implement LocalAgreement-2 for real-time partial results
- Extend contract to STREAM_EVENT_KEYS ⊃ CONTRACT_KEYS
- This is required for `src/pipeline/stream_processor.py` (Week 5-6)

### P7: Speaker Diarization (Optional)
- ECAPA-TDNN embeddings + clustering for multi-speaker lectures
- Integrate with ASR for speaker-attributed transcripts
- Evaluate only if lecture scenario includes Q&A or panel discussion

### P8: Full-Duplex Ideas (Steal from GPT-Live)
- Frame-level backchannel acknowledgment
- Preamble filler during LLM wait (Week 5 implementation target)
- Delegation: fast ASR → deep LLM with graceful recovery

---

## 6. Immediate Next Steps

1. **Install dependencies**: `faster-whisper`, `silero-vad`, `librosa`, `jiwer`
2. **Record test audio**: 5-10 Korean lecture utterances (clean + noisy + code-switched + silence)
3. **Run baseline**: faster-whisper large-v3-turbo on all 5 tracks
4. **Fill the matrix**: Measure CER, latency, VRAM, hallucination rate
5. **Select winner**: Choose the engine that fits the T4 + 500ms + Korean constraints
6. **Implement production module**: `src/audio/asr_engine.py` with the winning engine + VAD + confidence + postprocessing
