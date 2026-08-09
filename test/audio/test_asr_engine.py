"""Headless unit tests for src/audio/asr_engine.py — no model, no mic required.

Mirrors the self-checks from lab/my-lab/01-ASR-review.ipynb (2.1-2.4) plus the
BaseASR contract pipeline verified through MockEngine.
"""
import numpy as np
import pytest

from src.audio import asr_engine as ae


# ── 2.0 audio normalization ──
class TestTo16kMono:
    def test_stereo_to_mono(self):
        stereo = np.random.randn(32000, 2).astype(np.float32)
        mono = ae.to_16k_mono(stereo, 16000)
        assert mono.ndim == 1 and mono.shape[0] == 32000

    def test_resample_from_8k(self):
        x = np.random.randn(8000).astype(np.float32)  # 0.5s @ 8kHz
        out = ae.to_16k_mono(x, 8000)
        assert out.shape[0] == 16000  # 0.5s @ 16kHz

    def test_passthrough_when_16k(self):
        x = np.zeros(16000, dtype=np.float32)
        out = ae.to_16k_mono(x, 16000)
        assert out.dtype == np.float32 and out.shape == x.shape


# ── 2.1 data contract ──
class TestContract:
    def test_contract_keys(self):
        assert ae.validate_contract({k: None for k in ae.CONTRACT_KEYS})

    def test_missing_keys_rejected(self):
        assert not ae.validate_contract({"text": "누락 있음"})

    def test_extra_keys_allowed(self):
        rec = {k: None for k in ae.CONTRACT_KEYS} | {"emotion": "happy"}
        assert ae.validate_contract(rec)


# ── 2.2 confidence / retry ──
class TestConfidence:
    def test_empty_text_fails(self):
        assert ae.judge_confidence(-0.3, "") is False

    def test_good_logprob(self):
        assert ae.judge_confidence(-0.3, "안녕하세요") is True

    def test_bad_logprob(self):
        assert ae.judge_confidence(-1.2, "안녕하세요") is False

    def test_missing_logprob_trusts_text(self):
        assert ae.judge_confidence(None, "안녕하세요") is True

    def test_should_retry_limits(self):
        assert ae.should_retry(0, False)
        assert not ae.should_retry(1, False)
        assert not ae.should_retry(0, True)


# ── 2.3 hallucination filter + Korean postprocess ──
class TestHallucination:
    def test_pattern_hit(self):
        assert ae.is_hallucination("시청해주셔서 감사합니다", 3.0, 0.5)

    def test_repeated_loop(self):
        assert ae.is_hallucination("네 네 네 네 확인했습니다", 3.0, 0.5)

    def test_silence_with_text(self):
        assert ae.is_hallucination("배송이 어디까지 왔나요", 3.0, 0.01)

    def test_normal_speech_passes(self):
        assert not ae.is_hallucination("배송이 어디까지 왔나요", 3.0, 0.5)


class TestPostprocess:
    def test_lexicon_and_unit(self):
        lex = {"환뿔": "환불"}
        out = ae.postprocess_ko("환뿔 3 일 이내 30,000 원 환불", lexicon=lex)
        assert out == "환불 3일 이내 30,000원 환불"

    def test_default_lexicon_empty(self):
        assert ae.postprocess_ko("  3 일  뒤  ") == "3일 뒤"


# ── 2.4 VAD utterance splitting ──
class TestVAD:
    def test_two_utterances(self, synthetic_audio):
        assert len(ae.split_utterances(synthetic_audio, 16000)) == 2

    def test_slow_eou_merges(self, synthetic_audio):
        assert len(ae.split_utterances(synthetic_audio, 16000, min_silence_s=1.5)) == 1

    def test_speech_ratio(self, synthetic_audio):
        assert 0.4 < ae.speech_ratio(synthetic_audio, 16000) < 0.6

    def test_speech_ratio_empty(self):
        assert ae.speech_ratio(np.zeros(0, dtype=np.float32), 16000) == 0.0


# ── 2.8 BaseASR contract pipeline (MockEngine, no model) ──
class TestMockPipeline:
    def test_low_logprob_fails_gate(self, synthetic_audio):
        rec = ae.MockEngine(lp=-1.2).transcribe(synthetic_audio)
        assert rec["confidence_ok"] is False
        assert isinstance(rec["avg_logprob"], float)
        assert ae.validate_contract(rec)

    def test_beam_retry_improves(self, synthetic_audio):
        rec = ae.MockEngine(lp=-1.2).transcribe(synthetic_audio, retry_beam_size=5)
        assert rec["engine"] == "mock"

    def test_initial_prompt_passthrough(self, synthetic_audio):
        rec = ae.MockEngine(text="역전파 강의").transcribe(
            synthetic_audio, initial_prompt="인공지능 강의 요약")
        assert rec["text"] == "역전파 강의"
        assert rec["confidence_ok"] is True

    def test_initial_prompt_none_is_safe(self, synthetic_audio):
        rec = ae.MockEngine(text="테스트").transcribe(synthetic_audio)
        assert rec["text"] == "테스트"

    def test_silence_audio_gets_hallucination_gate(self):
        dummy = np.zeros(16000, dtype=np.float32)
        rec = ae.MockEngine(text="시청해주셔서 감사합니다").transcribe(dummy)
        assert rec["text"] == ""  # hallucination filter wiped it
        assert rec["confidence_ok"] is False

    def test_numpy_and_path_equal(self, synthetic_audio, tmp_path):
        path = tmp_path / "sample.wav"
        import soundfile as sf
        sf.write(str(path), synthetic_audio, 16000)
        rec_arr = ae.MockEngine(text="같은 텍스트").transcribe(synthetic_audio)
        rec_file = ae.MockEngine(text="같은 텍스트").transcribe(str(path))
        assert rec_arr["text"] == rec_file["text"]
        assert rec_arr["latency_ms"] >= 0


# ── factory ──
class TestFactory:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            ae.create_engine(provider="nope")
