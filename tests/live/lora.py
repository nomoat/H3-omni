"""Check exported LoRA projections against the released FP32 adapter."""
import argparse
import json
import struct
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pytorch'))
import torch
from native import Native

class Tensors:
    # Sequential reads avoid expensive mmap page faults on shared storage.
    def __init__(self,path):
        self.f=open(path,'rb'); self.start=8+struct.unpack('<Q',self.f.read(8))[0]
        self.meta=json.loads(self.f.read(self.start-8))
    def get(self,name):
        item=self.meta[name]; start,stop=item['data_offsets']
        self.f.seek(self.start+start)
        data=bytearray(self.f.read(stop-start))
        return torch.frombuffer(data,dtype={'BF16':torch.bfloat16,'F32':torch.float32}[item['dtype']]).reshape(item['shape'])

p=argparse.ArgumentParser()
p.add_argument('--adapter',default='/data0/models/h3-taomate/adapter/adapter_model.safetensors')
p.add_argument('--native',default='/data0/models/h3-taomate/native/FL2VA/transformer/taomate-lora.safetensors')
p.add_argument('--output',required=True)
a=p.parse_args(); torch.set_num_threads(4); torch.manual_seed(27)
source,target=Tensors(a.adapter),Tensors(a.native)
n=Native('build-cuda/libh3_cuda.so'); results=[]
for block in ('blocks.0','token_refiner.blocks.1'):
    for projection in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2'):
        prefix=block+'.'+projection
        ra=source.get(prefix+'.lora_a').bfloat16(); rb=source.get(prefix+'.lora_b').bfloat16()
        na=target.get(prefix+'.lora_a'); nb=target.get(prefix+'.lora_b')
        x=torch.randn(17,ra.shape[1]).bfloat16()
        expected=torch.nn.functional.linear(torch.nn.functional.linear(x.cuda(),ra.cuda()),rb.cuda()).cpu()
        xp,ap,bp=[n.tensor(t) for t in (x,na,nb)]
        mid=n.empty((17,128),torch.bfloat16); out=n.empty((17,nb.shape[0]),torch.bfloat16)
        n.call('lora_down_bf16',mid,xp,ap,17,na.shape[1])
        n.call('linear_bf16',out,mid,bp,None,17,128,nb.shape[0])
        actual=n.read(out,(17,nb.shape[0]),torch.bfloat16)
        if projection=='attn.qkv_proj':
            actual=actual.reshape(17,56,3,128).permute(0,2,1,3).reshape(17,21504)
        rel=float((actual.float()-expected.float()).norm()/expected.float().norm())
        row=dict(projection=prefix,relative_l2=rel,passed=rel<0.003)
        results.append(row);print(json.dumps(row),flush=True);n.clear()
# Preserve each separate FP32 operation in the sampler.
x=torch.randn(12345);v=torch.randn_like(x);sigma=0.81528664;next_sigma=0.375
sigma=torch.tensor(sigma).item();sigma_next=torch.tensor(next_sigma).item()
ratio=torch.tensor(sigma_next/sigma).item();one_minus=torch.tensor(1-ratio).item()
t=torch.tensor(1-torch.tensor(1-sigma).item()).item()
expected=ratio*x+one_minus*(x+t*v)
xp,vp=n.tensor(x),n.tensor(v)
n.call('taomate_step_f32',xp,vp,x.numel(),sigma,sigma_next)
actual=n.read(xp,x.shape,torch.float32)
results.append(dict(projection='GPU sampler FP32 order',passed=torch.equal(actual,expected)))
n.close()
Path(a.output).write_text(json.dumps(dict(passed=all(r['passed'] for r in results),cases=results),indent=2)+'\n')
assert all(r['passed'] for r in results)
