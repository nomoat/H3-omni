"""ctypes access to the independent C/CUDA backend, using the public C header."""
import ctypes as C
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


class Native:
    def __init__(self, library):
        self.lib = C.CDLL(str(Path(library).resolve()))
        source = (ROOT / "h3_gpu.h").read_text() + (ROOT / "cuda/h3_cuda.h").read_text()
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
        for result, name, args in re.findall(
            r"([\w ]+\s*\*?)\s*(h3_gpu_\w+)\s*\(([^;]+)\);", source
        ):
            fn = getattr(self.lib, name)
            fn.restype = self.ctype(result.strip())
            fn.argtypes = [self.ctype(a.strip().rsplit(" ", 1)[0] + (" *" if "*" in a.rsplit(" ", 1)[-1] else "")) for a in args.split(",") if a.strip() != "void"]
        self.owned = []
        error = C.create_string_buffer(512)
        self.gpu = self.lib.h3_gpu_create(None, error, len(error))
        if not self.gpu:
            raise RuntimeError(error.value.decode())

    @staticmethod
    def ctype(t):
        if "*" in t:
            return C.c_void_p
        return {"int": C.c_int, "float": C.c_float, "size_t": C.c_size_t,
                "uint32_t": C.c_uint32, "uint64_t": C.c_uint64,
                "h3_gpu_dtype": C.c_int, "void": None}[t]

    def call(self, name, *args):
        result = getattr(self.lib, "h3_gpu_" + name)(self.gpu, *args)
        if not result:
            p = self.lib.h3_gpu_error(self.gpu)
            raise RuntimeError(name + ": " + C.string_at(p).decode())
        return result

    def tensor(self, x):
        x = x.detach().cpu().contiguous()
        suffix = {torch.float32: "f32", torch.bfloat16: "bf16", torch.int32: "u32"}[x.dtype]
        p = self.call("tensor_from_" + suffix, x.data_ptr(), x.numel())
        self.owned.append(p)
        return p

    def empty(self, shape, dtype):
        return self.tensor(torch.zeros(shape, dtype=dtype))

    def read(self, p, shape, dtype):
        self.call("submit")
        x = torch.empty(shape, dtype=dtype)
        suffix = "bf16" if dtype == torch.bfloat16 else "f32"
        if not getattr(self.lib, "h3_gpu_tensor_read_" + suffix)(p, x.data_ptr(), x.numel()):
            raise RuntimeError("tensor read failed")
        return x

    def clear(self):
        for p in self.owned:
            self.lib.h3_gpu_tensor_free(p)
        self.owned.clear()

    def close(self):
        self.clear()
        self.lib.h3_gpu_free(self.gpu)
