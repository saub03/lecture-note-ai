# LLM Model Test Plan

> Synthesized from 16 reference notebooks in `lab/reference/LLM/` (2026-07-31)

---

## 1. Overview: The LLM Pipeline Architecture

Every notebook converges on the same engineering doctrine: **"Don't trust — verify"** (믿지 말고 검증한다). The LLM is not a black box but a contract-bounded component:

```
Input (ASR transcript + context) → [Prompt Contract: roles, budget, fencing]
     → [Model Load Ladder: fp16 → sanity probe → 4bit NF4 fallback]
     → [Guarded Generation: JSON mode → schema gate → retry → fallback]
     → [Output Contract: validated dict]
     → [Downstream: TTS / note formatting / stream processor]
```

### Models Evaluated (16 notebooks)

| Notebook | Model | Type | VRAM (T4) | Contract Compliance | Key Insight |
|---|---|---|---|---|---|
| 5-T | Mini Transformer (scratch) | Theory foundation | N/A | N/A | Causal mask = sequential generation constraint |
| 5-S | Upstage Solar (API + 10.7B local) | API + local | 10.7B: ~5.4GB (NF4) | 5/5, 0 retries | License table first; `reasoning_effort` dial |
| 5-P | Qwen3-0.6B + Mocks | Prompt engineering | N/A | Gate violation 60% → 10% | Prompts are contracts; gates guarantee |
| 5-A | Gemini 3.6 Flash (multimodal) | Cloud API single-call | N/A | 5/5, 0 retries | Catalog ≠ callable; `finish_reason` before parse |
| 5-D | DeepSeek-R1-Distill-Qwen-1.5B | Reasoning (AR) | ~3.6GB | Skip mode measured | TTFA 15.1s vs 0.2s (66× faster without think) |
| 5-E | Gemma 4 E2B | Local multimodal | ~10.2GB (fp16) | 5/5, 0 retries | fp16 overflow = silent garbage; sanity probe required |
| 5-G | Groq (Llama 3.3 70B, 3.1 8B) | Cloud API cascade | N/A | 5/5, 0 retries | Token speed IS UX; TTFT vs throughput |
| 5-K | EXAONE-4.0-1.2B | Local Korean | ~2.6GB (fp16) | 5/5, 0 retries | License gate as code; T=0.1 suppresses code-switching |
| 5-L | Llama-3.2-Korean-Bllossom-3B | Local Korean | ~6.4GB (fp16) | Validated output | LLM output is not a response until validated |
| 5-Q | Qwen3-8B | Local | ~5.4GB (NF4) | 5/5, 0 retries | Arithmetic eliminates fp16; thinking costs TTFA |
| 5-R | OpenRouter league (multi) | Cloud router | N/A | Per-model scores | Availability ≠ compliance; DNP tracked separately |
| 5-V | Gemma 4 E2B (audio input) | Local omni | ~10.2GB | 5/5, 0 retries | Data sovereignty; CER probe before contract loop |
| 7-O | Qwen2.5-Omni-3B | Local omni (ASR+LLM+TTS) | ~6GB (fp16) | Post-hoc only | Cost of omni: no intermediate transcript, no 2-pass guards |
| 8-T | Midm-2.0-Mini (QLoRA fine-tuned) | Local Korean | ~4-6GB | Pass rate improved | Format decides loss; training data = contract incarnate |
| async | N/A | Python machinery | N/A | N/A | Async overlaps waiting, not computing |
| decorator | N/A | Python machinery | N/A | N/A | Registration pattern; `functools.wraps` prevents metadata loss |

---

## 2. Key Metrics for LLM Model Evaluation

### 2.1 Latency Metrics (Voice-Agent Critical)

| Metric | Definition | Budget | Notes |
|---|---|---|---|
| **TTFT** (Time To First Token) | Request sent → first token received | Perceived responsiveness | Dominated by server queueing for APIs |
| **TTFA** (Time To First Answer) | Request → first *answer* token (after `</think>`) | ≤ 500ms for voice agents | Reasoning models: TTFA >> TTFT |
| **TTFS** (Time To First Sentence) | Request → first complete sentence | TTS handoff gate | More important than TTFT for TTS overlap |
| **TTFS − TTFT gap** | Time between first token and first sentence | Overlap budget | Gemma E2B: ~797ms; EXAONE 1.2B: ~72ms |
| **E2E Latency** | Input → final validated output | ≤ 2.5s per utterance | Includes ASR + LLM + validation |
| **Throughput** (tokens/s or chunks/s) | Sustained generation rate | Governs sentence-boundary overlap | Throughput > TTFT for streaming UX |

### 2.2 Contract Compliance Metrics

| Metric | Definition | Target |
|---|---|---|
| **Compliance rate** | Valid outputs / total attempts | 5/5 (100%) |
| **Retry count** | Retries needed per utterance | 0 (first-pass success) |
| **Schema violation rate** | Outputs failing jsonschema gate | 0% after prompt + few-shot |
| **No-JSON rate** | Outputs that aren't JSON at all | 0% |
| **Fence-stripping success** | Clean JSON extracted from markdown fences | 100% |

### 2.3 Quality Metrics

| Metric | Definition | How Measured |
|---|---|---|
| **CER** (Character Error Rate) | ASR transcript accuracy | Space-stripped Levenshtein vs reference |
| **Hangul ratio** | Korean character proportion in output | `hangul_chars / alpha_chars`; T=0.1 → ≥0.984 |
| **Hanja contamination** | CJK unified ideographs in Korean output | Flag if `hanja > 0` (Qwen3 tends to mix) |
| **Response length** | Character count vs contract limit | ≤ 200 chars (≈15s TTS) |
| **Parroting / OOD generalization** | Out-of-distribution utterance handling | Probe with 3 unseen scenarios |
| **Code-switching** | Foreign language mixing in Korean output | Hangul ratio + manual inspection |

### 2.4 Deployment Metrics

| Metric | Why It Matters | T4 Constraint |
|---|---|---|
| **VRAM (GB)** | Fits on T4 16GB | fp16: params × 2 bytes; NF4: params × 0.55 × 1.3 margin |
| **Model load time (s)** | Cold-start UX | Gemma E2B: ~10s; EXAONE 1.2B: ~3s |
| **Quantization tolerance** | Speed/accuracy tradeoff | NF4 buys VRAM, costs speed on small models |
| **Throughput at batch=1** | Single-item latency for real-time | Autoregressive penalty scales with output length |
| **License** | Commercial deployment legality | MIT ✓, CC-BY-NC ✗, qwen-research ✗, llama3.2 ✓ |

---

## 3. Processing Methods to Improve Performance

### 3.1 Prompt Engineering (Contract-Bounded)

| Method | Detail | Impact |
|---|---|---|
| **Role structure** | system → user → assistant; system exactly once, first; ends with user | Enforces template compliance |
| **Data fencing** | Customer utterance wrapped in `<고객발화>` tags declared as non-instructions | Injection resistance |
| **Token budget** | System + latest user inviolable; drop oldest middle turns first | Prevents silent truncation |
| **Few-shot anchoring** | 2-3 exemplar outputs matching target format | Gate violation 60% → 10% |
| **State-machine conditioning** | System prompt varies by dialog state (ASK_NEED/CONFIRM/PROCESS/CLOSE) | Context-appropriate responses |
| **TTS-safe output rules** | Digits as Korean readings, no symbols/emoji | Prevents TTS failures |
| **ASR confidence modulation** | Low CER → "don't guess — ask to confirm" instruction | Hallucination containment |

### 3.2 Output Validation (Guarded Generation Loop)

```
1. Generate with JSON mode (server) or prompt instruction (local)
2. Strip markdown fences
3. Validate against jsonschema
4. If invalid → retry with precise error note (name the missing key)
5. If still invalid → safe fallback response
6. Never throw — return status dict
```

| Layer | Purpose | Failure Mode |
|---|---|---|
| JSON mode / `response_format` | Guarantees "it is JSON" | Doesn't guarantee schema match |
| Prompt instruction | Encourages schema compliance | Probabilistic, not guaranteed |
| jsonschema gate | Validates structure | Catches what prompt missed |
| Retry with error note | Fixes specific missing keys | Burns latency budget |
| Safe fallback | Graceful degradation | Generic but valid response |

### 3.3 Model Selection Framework (Three Filters)

1. **License**: Commercial deployment legal? (MIT ✓, CC-BY-NC ✗, qwen-research ✗)
2. **VRAM arithmetic**: `params × 2 bytes (fp16)` or `params × 0.55 × 1.3 (NF4)` ≤ available VRAM
3. **Dependency check**: `transformers` version compatibility, `bitsandbytes` compute capability ≥ 7.5

**Elimination order**: License → VRAM math → dependency → probe → contract test

### 3.4 Load Ladder (Local Model Ops)

```
1. Attempt fp16 load
2. Sanity probe: non-empty output, Hangul present, no repetition collapse
3. If probe fails → downgrade to 4bit NF4 (compute fp16)
4. If still fails → loud stop with diagnostic
```

| Trap | Symptom | Fix |
|---|---|---|
| `bnb_4bit_quant_type` defaults to `"fp4"` | Silent quality loss | Explicitly set `"nf4"` |
| `bnb_4bit_compute_dtype` defaults to `float32` | Slow inference | Set `float16` |
| `device_map="auto"` leaves params on meta device | Silent partial load | Force `device_map={"": 0}`, assert placement |
| fp16 overflow on T4 (no bf16) | Garbage output (silent) | Sanity probe detects; NF4 fallback |
| Empty exception messages | `str(e)` == "" | Surface `type(e).__name__` + repr + traceback |

### 3.5 Reasoning Model Handling

| Mode | TTFT | TTFA | Use Case |
|---|---|---|---|
| **Natural (with think)** | ~2.3s | ~15.1s | Post-call batch: summaries, QA audits |
| **Skip (manual prefill)** | ~0.2s | ~0.2s | Real-time voice agents |

**Key insight**: Reasoning models are **disqualified from real-time voice loops** (customer hears thinking as silence) but valuable for **post-hoc batch work**. Skip-prefill is template-dependent and officially performance-degrading.

### 3.6 Streaming Architecture

| Component | Implementation | Purpose |
|---|---|---|
| `TextIteratorStreamer` + thread | Non-blocking token iteration | TTFT measurement + early TTS handoff |
| Sentence-boundary splitting | Split on `.?!` + Korean endings | TTS can start before full generation |
| TTFS−TTFT gap tracking | Measure overlap budget | Determines TTS pipeline design |
| `async for` on token stream | Process tokens on arrival | Perceived responsiveness |

### 3.7 Fine-Tuning Path (QLoRA)

| Aspect | Detail | Note |
|---|---|---|
| **Technique** | NF4 4-bit frozen base + fp16 LoRA adapters (r=16, α=32) | QLoRA ≈ 4-6GB vs full FT ≥18GB |
| **Data format** | Prompt-completion pairs; prompt tokens masked to −100 | Format decides the loss |
| **Validation** | All training examples pass contract validation before training | Never teach wrong answers |
| **Target modules** | q/k/v/o/gate/up/down projections | Standard LoRA coverage |
| **Reversibility** | Adapter detachment restores base model | Fine-tuning is reversible |
| **OOD probes** | 3 unseen scenarios to detect parroting | Small-set style tuning risks memorization |

### 3.8 Async/Streaming Python Machinery

| Pattern | Use Case | Key Rule |
|---|---|---|
| `asyncio.gather` | Parallel I/O waits (LLM + TTS + ASR) | Overlaps waiting, not computing |
| `asyncio.to_thread` | Blocking library in async context | `time.sleep` freezes the loop |
| `asyncio.wait_for` | Timeout ceiling on external calls | Prevents hung requests |
| `async for` on streams | Process tokens as they arrive | First audio out before full LLM response |
| Cancellation etiquette | Catch `CancelledError`, clean up, re-raise | Prevents zombie tasks |

### 3.9 Multimodal / Omni Models

| Architecture | Pros | Cons |
|---|---|---|
| **Cascade** (ASR → LLM → TTS) | Own intermediate transcript; 2-pass guards; swappable components | 2+ failure points; latency sum |
| **Cloud single-call** (Gemini) | 1 API call; simpler code | No intermediate transcript; vendor lock-in |
| **Local omni** (Qwen2.5-Omni) | Data sovereignty; no API calls | No 2-pass guards; sample rate bridging (16k→24k); Korean quality unverified |

**Cost of omni**: The model owns the system-prompt slot; there is no intermediate ASR transcript; **2-pass guard loops are structurally impossible** — the Talker consumes the Thinker's hidden states directly.

### 3.10 Provider / Router Resilience

| Guard | Implementation | Purpose |
|---|---|---|
| **Catalog guard** | `models.list()` + `supported_actions` filter | Listed ≠ callable |
| **Survival probe** | 1-token real call per candidate; 429 = alive | Detects dead models |
| **Rate gate** | `RateGate` with min interval per RPM limit | Attributable 429s |
| **Per-model capability degradation** | Track `response_format` support per model | JSON mode fallback |
| **DNP tracking** | Did-Not-Play (upstream saturation) separate from compliance failures | Availability ≠ quality |
| **Server-side fallback logging** | Compare requested model vs `resp.model` | Detect silent swaps |

---

## 4. Evaluation Protocol for lecture-note-ai

### 4.1 Test Tracks

| Track | Input Type | Metric | Purpose |
|---|---|---|---|
| **Clean Korean** | Well-formed lecture transcript | Compliance rate, latency | Baseline |
| **Noisy ASR** | Transcript with CER > 0.15 | Hallucination resistance | ASR-LLM integration |
| **Code-switched** | Korean + English technical terms | Hangul ratio, manual | Domain reality |
| **Injection attack** | Prompt injection in user input | Injection success rate | Security |
| **Streaming** | Simulated real-time chunks | TTFT, TTFS, TTFS−TTFT gap | TTS overlap budget |
| **Long context** | Full lecture (5000+ tokens) | Truncation strategy | Context window management |
| **Reasoning load** | Complex summarization task | TTFA vs TTFT | Reasoning model viability |

### 4.2 Model Selection Framework

Three filters (from 5-S, 5-K, 5-Q):

1. **License**: Commercial deployment legal? MIT ✓ / Apache 2.0 ✓ / CC-BY-NC ✗ / qwen-research ✗ / llama3.2 ✓
2. **VRAM arithmetic**: `params × 2 bytes (fp16)` or `params × 0.55 × 1.3 (NF4)` ≤ T4 16GB
3. **Ecosystem**: transformers v5 compatible, ungated (or HF token available), active community

### 4.3 Recommended Comparison Matrix

| Model | License | VRAM (fp16) | VRAM (NF4) | Compliance | TTFT (ms) | TTFS (ms) | Hangul Ratio | Notes |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-Korean-Bllossom-3B | llama3.2 ✓ | ~6.4GB | ~2.2GB | 5/5 | ? | ? | ? | Ungated, Korean full-tuned |
| EXAONE-4.0-1.2B | NC ✗ | ~2.6GB | ~1.0GB | 5/5 | ~200 | ~270 | ≥0.984 (T=0.1) | License blocks commercial |
| Gemma-4-E2B | Apache 2.0 ✓ | ~10.2GB | ~3.5GB | 5/5 | ~350 | ~1150 | ? | Gated, multimodal |
| Qwen3-8B | Apache 2.0 ✓ | ~16GB ✗ | ~5.4GB | 5/5 | ? | ? | ? | NF4 only on T4 |
| DeepSeek-R1-Distill-1.5B | MIT ✓ | ~3.6GB | ~1.3GB | ? | ~2300 | ? | ? | Reasoning model |
| Midm-2.0-Mini | MIT ✓ | ~4.6GB | ~1.7GB | ? | ? | ? | ? | Korean native, QLoRA-ready |
| Solar-10.7B | CC-BY-NC ✗ | ~21.4GB ✗ | ~5.4GB | 5/5 | ~1000 | ? | ? | License blocks commercial |

> **Fill `?` with actual measurements using the evaluation protocol above.**

---

## 5. Actionable Proposals (Priority Order)

### P0: Contract-First LLM Module
- Implement `src/llm/base.py` with `generate(prompt, context=None) -> dict` returning validated contract
- Implement `src/llm/factory.py` with license + VRAM + dependency gates
- JSON schema for summarization output: `{"summary": str, "key_points": list[str], "confidence": float}`
- Guarded generation loop: JSON mode → fence-strip → jsonschema → retry → fallback
- **Rationale**: Contract-first is the foundation; every downstream component depends on it

### P1: Prompt Contract + Injection Defense
- PROMPT_CONTRACT validator: roles, budget, data fencing
- Slide context injection as `<강의자료>` tags (non-instructions)
- ASR confidence modulation: low CER → "verify before summarizing" instruction
- Few-shot exemplars for note format
- **Rationale**: Prompts reduce violation rate; gates guarantee — both needed

### P2: Local Model Baseline (Llama-3.2-Korean-Bllossom-3B)
- Ungated, llama3.2 license, ~6.4GB fp16 on T4
- Load ladder: fp16 → sanity probe → NF4 fallback
- Measure TTFT, TTFS, compliance on 5 test tracks
- **Rationale**: Best balance of license, size, Korean support, and ecosystem

### P3: Streaming Integration
- `TextIteratorStreamer` + thread for TTFT measurement
- Sentence-boundary splitting for early TTS/note display
- TTFS−TTFT gap tracking → overlap budget for `src/pipeline/stream_processor.py`
- **Rationale**: Real-time lecture notes need streaming; TTFS gates downstream

### P4: Alternative Engine Evaluation
- **Qwen3-8B (NF4)**: Stronger reasoning, Apache 2.0, but NF4-only on T4
- **EXAONE-4.0-1.2B**: Fastest TTFT (~200ms), smallest VRAM, but NC license blocks commercial
- **Midm-2.0-Mini**: Korean native, MIT license, QLoRA-ready for style tuning
- **Gemma-4-E2B**: Multimodal (audio direct input), Apache 2.0, but gated + 10GB
- Swap via factory pattern → zero pipeline code change

### P5: Async Pipeline for Real-Time
- `asyncio.gather` for parallel ASR + LLM + TTS waits
- `asyncio.to_thread` for blocking Whisper inference
- `asyncio.wait_for` timeout ceiling on all external calls
- `async for` on token streams for progressive note display
- **Rationale**: Lecture notes must update in real-time; async overlaps waiting

### P6: Fine-Tuning for Lecture Style (QLoRA)
- Midm-2.0-Mini or Llama-3.2-Korean-Bllossom-3B as base
- 40-100 prompt-completion pairs matching target note format
- All training examples pre-validated against output contract
- QLoRA: NF4 base + fp16 adapters (r=16, α=32)
- OOD probes to detect parroting
- **Rationale**: Prompts + few-shot may not achieve consistent note style; fine-tuning shifts the prior

### P7: Reasoning Model for Post-Lecture Analysis
- DeepSeek-R1-Distill-1.5B for post-lecture deep summarization
- Natural mode (with thinking) for quality; skip mode for speed comparison
- Not for real-time — batch after lecture ends
- **Rationale**: Reasoning models excel at complex summarization but are disqualified from real-time loops

### P8: Multimodal Single-Call (Optional)
- Gemma-4-E2B audio direct input: cascade-free ASR+LLM
- Requires transcript as explicit contract field for verifiability
- Data sovereignty: real lecture audio never leaves the process
- **Rationale**: Simplifies architecture but loses 2-pass guards and intermediate transcript

---

## 6. Immediate Next Steps

1. **Install dependencies**: `transformers>=5.14`, `bitsandbytes`, `accelerate`, `jsonschema`, `huggingface_hub`
2. **HF token setup**: For gated models (Gemma, Llama) — store in `.env`
3. **Run baseline**: Llama-3.2-Korean-Bllossom-3B on 5 test tracks
4. **Fill the matrix**: Measure VRAM, TTFT, TTFS, compliance, Hangul ratio
5. **Implement contract module**: `src/llm/base.py` + `src/llm/factory.py` with guarded generation
6. **Integrate with ASR**: Connect `src/audio/asr_engine.py` output → LLM input with confidence modulation
7. **Test streaming**: Measure TTFS−TTFT gap for TTS/note overlap budget
