"""Embedded NLLB-200 translation engine.

The model is loaded lazily from a ModelScope snapshot on the translation
worker thread.  Keeping this separate from nllb_engine.py preserves the
existing HTTP nllb-serve integration.
"""
from __future__ import annotations

import threading

from .base import Translator, TranslatorError


_NLLB_LANG_MAP = {
    "auto": "",
    "en": "eng_Latn",
    "en-us": "eng_Latn",
    "en-gb": "eng_Latn",
    "zh": "zho_Hans",
    "zh-hans": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh-hant": "zho_Hant",
    "ja": "jpn_Jpan",
    "ja-jp": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
}


def _to_nllb_code(lang: str) -> str:
    key = (lang or "").strip().lower()
    return _NLLB_LANG_MAP.get(key, lang.strip() if lang else "")


class NllbLocalTranslator(Translator):
    """Run NLLB-200 directly in the application process."""

    serial = True

    def __init__(self, cfg, source_lang: str = "auto", target_lang: str = "zh-Hans"):
        super().__init__(cfg, source_lang, target_lang)
        self._src_code = _to_nllb_code(source_lang)
        self._tgt_code = _to_nllb_code(target_lang)
        if not self._src_code:
            raise TranslatorError(
                "内置 NLLB-200 不支持自动检测源语言，请在翻译设置中选择具体源语言。"
            )
        if not self._tgt_code:
            raise TranslatorError("NLLB-200 目标语言不能为空。")

        self._model_id = (
            getattr(cfg, "nllb_local_model", "facebook/nllb-200-distilled-600M")
            or "facebook/nllb-200-distilled-600M"
        ).strip()
        self._device_request = (
            getattr(cfg, "nllb_local_device", "auto") or "auto"
        ).strip().lower()
        self._dtype_request = (
            getattr(cfg, "nllb_local_dtype", "auto") or "auto"
        ).strip().lower()
        self._max_source_tokens = max(
            32, int(getattr(cfg, "nllb_local_max_source_tokens", 512) or 512)
        )
        self._max_new_tokens = max(
            8, int(getattr(cfg, "nllb_local_max_new_tokens", 128) or 128)
        )
        self._num_beams = max(1, int(getattr(cfg, "nllb_local_num_beams", 2) or 2))

        self._load_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                from ..asr._device import resolve_device
                from ..asr.modelscope_hub import download_modelscope

                requested = "cuda" if self._device_request == "auto" else self._device_request
                device = resolve_device(requested)
                dtype = self._resolve_dtype(torch, device)
                model_path = download_modelscope(self._model_id, "NLLB-200")

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                language_ids = getattr(tokenizer, "lang_code_to_id", {}) or {}
                if language_ids and self._src_code not in language_ids:
                    raise TranslatorError(
                        f"NLLB-200 不支持源语言码：{self._src_code}"
                    )
                tokenizer.src_lang = self._src_code
                target_id = language_ids.get(self._tgt_code)
                if target_id is None:
                    target_id = tokenizer.convert_tokens_to_ids(self._tgt_code)
                if target_id is None or target_id == tokenizer.unk_token_id:
                    raise TranslatorError(
                        f"NLLB-200 不支持语言码：{self._src_code} -> {self._tgt_code}"
                    )

                model_kwargs = {}
                if dtype is not None:
                    model_kwargs["dtype"] = dtype
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path, **model_kwargs
                )
                model.to(device)
                model.eval()
            except TranslatorError:
                raise
            except ImportError as error:
                raise TranslatorError(
                    "内置 NLLB-200 需要 transformers、sentencepiece、safetensors 和 torch，"
                    "请先安装本地翻译依赖。"
                ) from error
            except Exception as error:
                raise TranslatorError(f"NLLB-200 模型加载失败：{error}") from error

            self._torch = torch
            self._device = device
            self._tokenizer = tokenizer
            self._model = model

    def _resolve_dtype(self, torch, device):
        requested = self._dtype_request
        if requested in {"auto", ""}:
            requested = "float16" if device == "cuda" else "float32"
        if device == "cpu" and requested == "float16":
            requested = "float32"
        if requested == "float16":
            return torch.float16
        if requested == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _do_translate(self, text: str) -> str:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        if tokenizer is None or model is None or torch is None or self._device is None:
            raise TranslatorError("NLLB-200 模型尚未准备完成。")

        try:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_source_tokens,
            )
            encoded = {
                key: value.to(self._device) for key, value in encoded.items()
            }
            language_ids = getattr(tokenizer, "lang_code_to_id", {}) or {}
            target_id = language_ids.get(self._tgt_code)
            if target_id is None:
                target_id = tokenizer.convert_tokens_to_ids(self._tgt_code)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=target_id,
                    max_new_tokens=self._max_new_tokens,
                    num_beams=self._num_beams,
                    do_sample=False,
                )
            result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            if not result:
                raise TranslatorError("NLLB-200 返回空译文。")
            return result
        except TranslatorError:
            raise
        except Exception as error:
            raise TranslatorError(f"NLLB-200 翻译失败：{error}") from error

    def close(self) -> None:
        model = self._model
        torch = self._torch
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        if model is not None:
            del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
