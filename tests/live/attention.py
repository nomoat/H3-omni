"""Rectangular live attention vs independent PyTorch math, including offsets."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pytorch'))
import torch
from native import Native

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
n = Native('build-cuda/libh3_cuda.so')
torch.manual_seed(20261012)
cases = []
for heads in (28, 56):
    for text, media, history in ((7, 19, 0), (23, 263, 517), (128, 129, 131)):
        q = torch.randn(text + media + 3, heads, 128).bfloat16()
        k, v = [torch.randn(text + media + 3, heads, 128).bfloat16() for _ in range(2)]
        hk, hv = [torch.randn(history, heads, 128).bfloat16() for _ in range(2)]
        ck = torch.cat((k[1:1+text], hk, k[1+text:1+text+media]))
        cv = torch.cat((v[1:1+text], hv, v[1+text:1+text+media]))
        qp, kp, vp, ckp, cvp = [n.tensor(x) for x in (q,k,v,ck,cv)]
        out = n.empty(q.shape, torch.bfloat16)
        n.call('cross_attention_bf16', out, 1, qp, 1, kp, vp, 1, text, text, heads)
        n.call('cross_attention_bf16', out, 1+text, qp, 1+text, ckp, cvp, 0,
               media, text+media+history, heads)
        actual = n.read(out, q.shape, torch.bfloat16).float()[1:1+text+media]
        def att(q,k,v):
            return torch.nn.functional.scaled_dot_product_attention(
                q.float().transpose(0,1), k.float().transpose(0,1), v.float().transpose(0,1)).transpose(0,1)
        expected = torch.cat((att(q[1:1+text],k[1:1+text],v[1:1+text]),
                              att(q[1+text:1+text+media],ck,cv)))
        diff = actual-expected
        rel = float(diff.norm()/expected.norm())
        record = dict(heads=heads,text=text,media=media,history=history,
                      relative_l2=rel,max_abs=float(diff.abs().max()),
                      passed=rel<0.004 and torch.isfinite(actual).all().item())
        cases.append(record)
        print(json.dumps(record), flush=True)
        n.clear()
n.close()
Path(a.output).write_text(json.dumps(dict(passed=all(x['passed'] for x in cases),cases=cases),indent=2)+'\n')
assert all(x['passed'] for x in cases)
