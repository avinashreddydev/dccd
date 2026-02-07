from abc import ABC, abstractmethod
from typing import Dict, List
from pydantic import BaseModel
from dataclasses import dataclass
from vllm import LLM
from tokenizers import Tokenizer
import torch
import gc

class BaseAlgorithm(ABC):
    
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self._closed = False
    
    def close(self):
        """Best-effort release of GPU + worker resources."""
        if self._closed:
            return
        self._closed = True

        # Drop references first
        try:
            self.tokenizer = None
        except Exception:
            pass

        llm = getattr(self, "llm", None)
        self.llm = None

        # vLLM may have internal executors/workers; try common cleanup hooks
        try:
            if llm is not None:
                # Some versions expose internal engine/executor objects
                engine = getattr(llm, "llm_engine", None)
                if engine is not None:
                    executor = getattr(engine, "executor", None)

                    # Try a few possible shutdown/destroy method names
                    for obj in (executor, engine, llm):
                        for m in ("shutdown", "close", "destroy", "stop"):
                            fn = getattr(obj, m, None)
                            if callable(fn):
                                try:
                                    fn()
                                except Exception:
                                    pass
        finally:
            # Force Python to collect anything now-unreferenced
            gc.collect()

            # Clear CUDA caching allocator (helps reduce "reserved" memory)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Optional: can reduce fragmentation in some workloads
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don't suppress exceptions

    def __del__(self):
        # Fallback only; do NOT rely on this for correctness
        try:
            self.close()
        except Exception:
            pass