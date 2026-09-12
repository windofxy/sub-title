"""Qwen3-ASR segment engine. Native streaming is available separately through vLLM."""
from __future__ import annotations

import importlib.util
import logging
import time
from collections import deque

import numpy as np

from .base import AsrEngine, OnResult
from .modelscope_hub import download_modelscope

logger = logging.getLogger(__name__)


_SAMPLE_RATE = 16000
_SENTENCE_END = frozenset("。！？!?；;")
_COMMITTED_TAIL_CHARS = 256


def _clean_text(text: str) -> str:
    """Normalize model text enough for stable-prefix comparisons."""
    return " ".join((text or "").split())


def _is_ascii_word_char(char: str) -> bool:
    return char.isascii() and char.isalnum()


def _longest_common_prefix(texts: list[str]) -> str:
    if not texts:
        return ""
    limit = min(len(text) for text in texts)
    for index in range(limit):
        char = texts[0][index]
        if any(text[index] != char for text in texts[1:]):
            return texts[0][:index]
    return texts[0][:limit]


def _trim_to_safe_boundary(text: str) -> str:
    """Keep a stable prefix only when it ends at a sentence or word boundary."""
    value = text.rstrip()
    if not value:
        return ""
    if value[-1] in _SENTENCE_END:
        return value
    for index in range(len(value) - 1, -1, -1):
        if value[index] in _SENTENCE_END:
            return value[:index + 1]
    for index in range(len(value) - 1, -1, -1):
        if value[index].isspace():
            return value[:index].rstrip()
    # A non-CJK hypothesis ending at an ASCII word character has a complete word.
    if _is_ascii_word_char(value[-1]) and not any(
        "\u4e00" <= char <= "\u9fff" for char in value
    ):
        return value
    return ""


def qwen3_asr_available() -> bool:
    """Check availability without importing the heavyweight model runtime."""
    return importlib.util.find_spec("qwen_asr") is not None


def _patch_qwen_audio_conv_dtype(model) -> None:
    """Keep Qwen3-ASR audio features compatible with the first audio conv."""
    base_model = getattr(model, "model", None)
    thinker = getattr(base_model, "thinker", None)
    audio_tower = getattr(thinker, "audio_tower", None)
    conv = getattr(audio_tower, "conv2d1", None)
    register_hook = getattr(conv, "register_forward_pre_hook", None)
    bias = getattr(conv, "bias", None)
    target_dtype = getattr(bias, "dtype", None)
    if target_dtype is None:
        target_dtype = getattr(getattr(conv, "weight", None), "dtype", None)
    if not callable(register_hook) or target_dtype is None:
        logger.warning("Qwen3-ASR audio dtype adapter could not be installed")
        return

    def cast_audio_features(module, args):
        if not args:
            return args
        audio_features = args[0]
        to = getattr(audio_features, "to", None)
        if not callable(to):
            return args
        return (to(dtype=getattr(module.bias, "dtype", target_dtype)), *args[1:])

    handle = register_hook(cast_audio_features)
    setattr(model, "_subtitle_qwen_audio_dtype_hook", handle)
    logger.info("Enabled Qwen3-ASR audio convolution dtype adapter (%s)", target_dtype)


class Qwen3AsrEngine(AsrEngine):
    def __init__(self, cfg, on_result: OnResult, source: str = "system"):
        super().__init__(cfg, on_result, source=source)
        self.model = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._segment_samples = 32000
        self._overlap_samples = 6400
        self._endpoint_silence_samples = 8000
        self._punctuation_silence_samples = 4800
        self._max_window_samples = 96000
        self._min_endpoint_samples = 8000
        self._stable_history_size = 3
        self._punctuation_confirmations = 2
        self._stable_prefix_min_chars = 4
        self._silence_threshold = 0.01
        self._silence_run = 0
        self._speech_samples = 0
        self._samples_since_infer = 0
        self._hypotheses: deque[str] = deque(maxlen=self._stable_history_size)
        self._committed_tail = ""
        self._last_partial = ""
        self._inference_count = 0
        self._slow_inference_count = 0
        self._total_inference_seconds = 0.0
        self._max_inference_seconds = 0.0
        self._vad_enabled = False
        self._vad_model = None
        self._vad_torch = None
        self._vad_frame_samples = 512
        self._vad_frame_buf = np.zeros(0, dtype=np.float32)
        self._vad_context = np.zeros(0, dtype=np.float32)
        self._vad_active = False
        self._vad_speech_run = 0
        self._vad_silence_run = 0
        self._vad_start_threshold = 0.60
        self._vad_end_threshold = 0.30
        self._vad_start_samples = 3200
        self._vad_end_samples = 8000
        self._vad_pre_roll_samples = 4000
        self._vad_post_roll_samples = 4000
        self._vad_fallback_logged = False
        self._closed = False

    def load(self) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as error:
            import platform
            hint = (
                "运行 scripts\\install_qwen3_asr.bat"
                if platform.system() == "Windows"
                else "pip install qwen-asr"
            )
            raise ImportError(
                f"Qwen3-ASR 未安装。{hint}，"
                "或直接执行：pip install qwen-asr"
            ) from error
        model_id = getattr(self.cfg, "qwen3_asr_model", "Qwen/Qwen3-ASR-0.6B")
        # 跨平台设备解析：cuda 不可用（macOS/CPU torch）时降级，避免硬崩。
        from ._device import resolve_device, cuda_available
        device = resolve_device(getattr(self.cfg, "qwen3_asr_device", "cuda"))
        seconds = float(getattr(self.cfg, "qwen3_asr_segment_seconds", 2.0))
        quantization = getattr(self.cfg, "qwen3_asr_quantization", "none")
        self._configure_windowing(seconds)
        if quantization not in {"none", "4bit"}:
            raise ValueError("Qwen3-ASR 量化模式仅支持 none 或 4bit")
        if quantization == "4bit":
            # bitsandbytes 仅 CUDA 可用（无 macOS wheel，MPS/CPU 都不支持）。
            if device != "cuda" or not cuda_available():
                raise ValueError(
                    "Qwen3-ASR 4bit 量化仅支持 CUDA；CPU/MPS 或无 CUDA 时请用原始精度（none）"
                    "或 faster-whisper INT8。"
                )
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as error:
                raise ImportError(
                    "Qwen3-ASR 4bit 量化需要 bitsandbytes。请执行：pip install bitsandbytes"
                ) from error
        # MPS 上 Qwen3 用 bfloat16 可能不稳，统一 cpu 用 float32、其余（cuda/mps）bf16。
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        device_map = device if device == "cpu" else f"{device}:0"
        precision = "4bit" if quantization == "4bit" else str(dtype).replace("torch.", "")
        logger.info(f"从 ModelScope 下载并加载 {model_id} (device={device}, {precision})...")
        model_path = download_modelscope(model_id, "Qwen3-ASR")
        model_kwargs = dict(
            dtype=dtype, device_map=device_map,
            max_inference_batch_size=1, max_new_tokens=128,
        )
        if quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        self.model = Qwen3ASRModel.from_pretrained(
            model_path, **model_kwargs,
        )
        if quantization == "4bit":
            _patch_qwen_audio_conv_dtype(self.model)
        self._load_vad()
        logger.info(
            "Qwen3-ASR ready (cadence=%.2fs, overlap=%.2fs, endpoint=%.2fs, max_window=%.2fs, vad=%s)",
            self._segment_samples / _SAMPLE_RATE,
            self._overlap_samples / _SAMPLE_RATE,
            self._endpoint_silence_samples / _SAMPLE_RATE,
            self._max_window_samples / _SAMPLE_RATE,
            self._vad_model is not None,
        )

    def _configure_windowing(self, cadence_seconds: float) -> None:
        cadence_seconds = max(0.1, cadence_seconds)
        overlap_seconds = max(
            0.0, float(getattr(self.cfg, "qwen3_asr_overlap_seconds", 0.4))
        )
        endpoint_seconds = max(
            0.05, float(getattr(self.cfg, "qwen3_asr_endpoint_silence_seconds", 0.5))
        )
        punctuation_seconds = max(
            0.05,
            float(getattr(self.cfg, "qwen3_asr_punctuation_silence_seconds", 0.3)),
        )
        max_window_seconds = max(
            cadence_seconds,
            float(getattr(self.cfg, "qwen3_asr_max_window_seconds", 6.0)),
        )
        self._segment_samples = max(1600, int(cadence_seconds * _SAMPLE_RATE))
        self._overlap_samples = max(0, int(overlap_seconds * _SAMPLE_RATE))
        self._endpoint_silence_samples = max(1, int(endpoint_seconds * _SAMPLE_RATE))
        self._punctuation_silence_samples = max(
            1, int(punctuation_seconds * _SAMPLE_RATE)
        )
        self._max_window_samples = max(
            self._segment_samples, int(max_window_seconds * _SAMPLE_RATE)
        )
        self._min_endpoint_samples = max(1600, min(self._segment_samples, 8000))
        self._stable_history_size = max(
            2, int(getattr(self.cfg, "qwen3_asr_stable_history_size", 3))
        )
        self._punctuation_confirmations = max(
            1, int(getattr(self.cfg, "qwen3_asr_punctuation_confirmations", 2))
        )
        self._stable_prefix_min_chars = max(
            1, int(getattr(self.cfg, "qwen3_asr_stable_prefix_min_chars", 4))
        )
        self._vad_enabled = bool(getattr(self.cfg, "qwen3_asr_vad_enabled", False))
        self._vad_start_threshold = min(
            1.0, max(0.0, float(getattr(self.cfg, "qwen3_asr_vad_start_threshold", 0.60)))
        )
        self._vad_end_threshold = min(
            self._vad_start_threshold,
            max(0.0, float(getattr(self.cfg, "qwen3_asr_vad_end_threshold", 0.30))),
        )
        self._vad_start_samples = max(
            self._vad_frame_samples,
            int(max(0.0, float(getattr(self.cfg, "qwen3_asr_vad_start_seconds", 0.20))) * _SAMPLE_RATE),
        )
        self._vad_end_samples = max(
            self._vad_frame_samples,
            int(max(0.0, float(getattr(self.cfg, "qwen3_asr_vad_end_seconds", 0.50))) * _SAMPLE_RATE),
        )
        self._vad_pre_roll_samples = max(
            self._vad_frame_samples,
            int(max(0.0, float(getattr(self.cfg, "qwen3_asr_vad_pre_roll_seconds", 0.25))) * _SAMPLE_RATE),
        )
        self._vad_post_roll_samples = max(
            self._vad_frame_samples,
            int(max(0.0, float(getattr(self.cfg, "qwen3_asr_vad_post_roll_seconds", 0.25))) * _SAMPLE_RATE),
        )
        self._hypotheses = deque(maxlen=self._stable_history_size)

    def _load_vad(self) -> None:
        self._vad_model = None
        self._vad_torch = None
        self._vad_fallback_logged = False
        if not self._vad_enabled:
            return
        try:
            import torch
            from silero_vad import load_silero_vad

            self._vad_torch = torch
            self._vad_model = load_silero_vad(onnx=False)
            eval_method = getattr(self._vad_model, "eval", None)
            if callable(eval_method):
                eval_method()
            logger.info(
                "Silero VAD enabled (start=%.2f, end=%.2f, end_silence=%.2fs)",
                self._vad_start_threshold,
                self._vad_end_threshold,
                self._vad_end_samples / _SAMPLE_RATE,
            )
        except Exception as error:
            self._vad_model = None
            self._vad_torch = None
            logger.warning(
                "Silero VAD unavailable; falling back to RMS silence detection: %s",
                error,
            )

    def feed(self, chunk: np.ndarray) -> None:
        if self._closed or self.model is None:
            return
        chunk = chunk.astype(np.float32, copy=False)
        if self._vad_model is not None:
            try:
                self._feed_vad(chunk)
            except Exception as error:
                self._vad_model = None
                self._vad_torch = None
                self._vad_frame_buf = np.zeros(0, dtype=np.float32)
                if not self._vad_fallback_logged:
                    logger.warning(
                        "Silero VAD inference failed; falling back to RMS silence detection: %s",
                        error,
                    )
                    self._vad_fallback_logged = True
                self._reset_window()
                self._feed_rms(chunk)
            return

        self._feed_rms(chunk)

    def _feed_rms(self, chunk: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, chunk])
        self._samples_since_infer += len(chunk)
        energy = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        if energy < self._silence_threshold:
            self._silence_run += len(chunk)
        else:
            self._silence_run = 0
            self._speech_samples += len(chunk)
        if self._should_finalize_from_silence():
            self._samples_since_infer = 0
            self._infer_window(finalize=True)
        elif self._samples_since_infer >= self._segment_samples:
            self._samples_since_infer = 0
            self._infer_window()

    def _feed_vad(self, chunk: np.ndarray) -> None:
        self._vad_frame_buf = np.concatenate([self._vad_frame_buf, chunk])
        while len(self._vad_frame_buf) >= self._vad_frame_samples:
            frame = self._vad_frame_buf[: self._vad_frame_samples]
            self._vad_frame_buf = self._vad_frame_buf[self._vad_frame_samples:]
            context_limit = max(
                self._vad_frame_samples,
                self._vad_pre_roll_samples,
                self._vad_start_samples,
            )
            self._vad_context = np.concatenate([self._vad_context, frame])[-context_limit:]
            probability = self._vad_probability(frame)

            if not self._vad_active:
                if probability >= self._vad_start_threshold:
                    self._vad_speech_run += len(frame)
                else:
                    self._vad_speech_run = 0
                if self._vad_speech_run < self._vad_start_samples:
                    continue
                self._vad_active = True
                self._buf = self._vad_context.copy()
                self._speech_samples = len(self._buf)
                self._samples_since_infer = 0
                self._silence_run = 0
                self._vad_silence_run = 0
                continue

            self._buf = np.concatenate([self._buf, frame])
            self._speech_samples += len(frame)
            self._samples_since_infer += len(frame)
            if probability <= self._vad_end_threshold:
                self._vad_silence_run += len(frame)
            else:
                self._vad_silence_run = 0

            required_silence = max(self._vad_end_samples, self._vad_post_roll_samples)
            if self._vad_silence_run >= required_silence:
                self._vad_active = False
                self._vad_speech_run = 0
                self._vad_silence_run = 0
                self._samples_since_infer = 0
                if len(self._buf) > self._min_endpoint_samples:
                    self._infer_window(finalize=True)
                else:
                    self._reset_window()
            elif self._samples_since_infer >= self._segment_samples:
                self._samples_since_infer = 0
                self._infer_window()

    def _vad_probability(self, frame: np.ndarray) -> float:
        if self._vad_model is None or self._vad_torch is None:
            return 0.0
        audio = self._vad_torch.from_numpy(frame.copy())
        inference_mode = getattr(self._vad_torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else self._vad_torch.no_grad()
        with context:
            output = self._vad_model(audio, _SAMPLE_RATE)
        reshape = getattr(output, "reshape", None)
        if callable(reshape):
            output = reshape(-1)[-1]
        item = getattr(output, "item", None)
        return float(item() if callable(item) else output)

    def _should_finalize_from_silence(self) -> bool:
        if self._speech_samples == 0 or len(self._buf) < self._min_endpoint_samples:
            return False
        if self._silence_run >= self._endpoint_silence_samples:
            return True
        return (
            self._silence_run >= self._punctuation_silence_samples
            and self._has_confirmed_terminal_punctuation()
        )

    def _has_confirmed_terminal_punctuation(self) -> bool:
        if len(self._hypotheses) < self._punctuation_confirmations:
            return False
        recent = list(self._hypotheses)[-self._punctuation_confirmations:]
        return all(text.rstrip().endswith(tuple(_SENTENCE_END)) for text in recent)

    def _infer_window(self, *, finalize: bool = False) -> None:
        if self._speech_samples == 0:
            self._reset_window()
            return
        audio_seconds = len(self._buf) / _SAMPLE_RATE
        started_at = time.perf_counter()
        try:
            language = getattr(self.cfg, "qwen3_asr_language", "Chinese")
            result = self.model.transcribe(
                audio=(self._buf, _SAMPLE_RATE), language=language
            )
            raw_text = _clean_text(result[0].text if result else "")
            text = self._remove_committed_overlap(raw_text)
            if finalize:
                final_text = text or self._last_partial
                if final_text:
                    self._emit_final(final_text)
                self._reset_window()
                return

            if text:
                self._hypotheses.append(text)
                if len(self._buf) >= self._max_window_samples:
                    stable_prefix = self._stable_prefix()
                    if stable_prefix:
                        self._roll_window(stable_prefix, text)
                        return
                self._emit_partial(text)
        except Exception as error:
            logger.exception("Qwen3-ASR inference failed: %s", error)
        finally:
            self._record_inference_time(time.perf_counter() - started_at, audio_seconds)

    def _record_inference_time(self, elapsed: float, audio_seconds: float) -> None:
        self._inference_count += 1
        self._total_inference_seconds += elapsed
        self._max_inference_seconds = max(self._max_inference_seconds, elapsed)
        logger.debug(
            "Qwen3-ASR inference %.2fs for %.2fs audio (call=%d)",
            elapsed,
            audio_seconds,
            self._inference_count,
        )
        cadence_seconds = self._segment_samples / _SAMPLE_RATE
        if elapsed <= cadence_seconds:
            return
        self._slow_inference_count += 1
        if self._slow_inference_count == 1 or self._slow_inference_count % 10 == 0:
            logger.warning(
                "Qwen3-ASR inference is slower than its %.2fs cadence: "
                "elapsed=%.2fs, slow_calls=%d; audio capture may queue or drop blocks",
                cadence_seconds,
                elapsed,
                self._slow_inference_count,
            )

    def _stable_prefix(self) -> str:
        if len(self._hypotheses) < self._stable_history_size:
            return ""
        prefix = _trim_to_safe_boundary(_longest_common_prefix(list(self._hypotheses)))
        if len(prefix) < self._stable_prefix_min_chars:
            return ""
        return prefix

    def _remove_committed_overlap(self, text: str) -> str:
        if not text or not self._committed_tail:
            return text
        max_size = min(len(self._committed_tail), len(text))
        for size in range(max_size, 1, -1):
            if self._committed_tail[-size:] != text[:size]:
                continue
            old_boundary = len(self._committed_tail) - size
            if self._splits_ascii_word(self._committed_tail, old_boundary):
                continue
            if self._splits_ascii_word(text, size):
                continue
            return text[size:].lstrip()
        return text

    @staticmethod
    def _splits_ascii_word(text: str, boundary: int) -> bool:
        return (
            0 < boundary < len(text)
            and _is_ascii_word_char(text[boundary - 1])
            and _is_ascii_word_char(text[boundary])
        )

    def _emit_partial(self, text: str) -> None:
        self._last_partial = text
        self.on_result(text, is_final=False, source=self.source, spk_id=None)

    def _emit_final(self, text: str) -> None:
        self.on_result(text, is_final=True, source=self.source, spk_id=None)
        self._committed_tail = (self._committed_tail + text)[-_COMMITTED_TAIL_CHARS:]

    def _roll_window(self, stable_prefix: str, hypothesis: str) -> None:
        self._emit_final(stable_prefix)
        suffix = hypothesis[len(stable_prefix):].lstrip()
        stable_fraction = min(1.0, len(stable_prefix) / max(1, len(hypothesis)))
        estimated_boundary = int(len(self._buf) * stable_fraction)
        keep_from = max(0, estimated_boundary - self._overlap_samples)
        self._buf = self._buf[keep_from:].copy()
        self._silence_run = 0
        self._speech_samples = len(self._buf)
        self._samples_since_infer = 0
        self._hypotheses.clear()
        self._last_partial = ""
        if suffix:
            self._emit_partial(suffix)

    def _reset_window(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._silence_run = 0
        self._speech_samples = 0
        self._samples_since_infer = 0
        self._hypotheses.clear()
        self._committed_tail = ""
        self._last_partial = ""
        self._vad_active = False
        self._vad_speech_run = 0
        self._vad_silence_run = 0
        self._vad_context = np.zeros(0, dtype=np.float32)
        reset_states = getattr(self._vad_model, "reset_states", None)
        if callable(reset_states):
            reset_states()

    def _flush_vad_tail(self) -> None:
        if self._vad_model is None or not self._vad_active or not len(self._vad_frame_buf):
            return
        self._buf = np.concatenate([self._buf, self._vad_frame_buf])
        self._speech_samples += len(self._vad_frame_buf)
        self._vad_frame_buf = np.zeros(0, dtype=np.float32)

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._flush_vad_tail()
        if self.model is not None and len(self._buf) > 1600:
            self._infer_window(finalize=True)
        else:
            self._reset_window()
        if self._inference_count:
            logger.info(
                "Qwen3-ASR inference summary: calls=%d, avg=%.2fs, max=%.2fs, slow=%d",
                self._inference_count,
                self._total_inference_seconds / self._inference_count,
                self._max_inference_seconds,
                self._slow_inference_count,
            )

    def reset(self) -> None:
        self._reset_window()
        self._vad_frame_buf = np.zeros(0, dtype=np.float32)
        self._closed = False
