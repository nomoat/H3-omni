"""Check whether native projections and rectangular attention preserve sharding."""
import argparse,json,sys,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pytorch'))
import torch
from native import Native
p=argparse.ArgumentParser()
p.add_argument('--adapter',type=Path,default=Path('/data0/models/h3-taomate/native/FL2VA/transformer/taomate-lora.safetensors'))
p.add_argument('--output',type=Path)
args=p.parse_args()
n=Native('build-cuda/libh3_cuda.so');torch.set_num_threads(4);torch.manual_seed(29)
failures=[];results=[]
path=args.adapter
f=path.open('rb');off=8+struct.unpack('<Q',f.read(8))[0];meta=json.loads(f.read(off-8))
def tensor(name):
    v=meta[name];a,b=v['data_offsets'];f.seek(off+a)
    return torch.frombuffer(bytearray(f.read(b-a)),dtype=torch.bfloat16).reshape(v['shape'])
def compare(name,a,b):
    d=a.float()-b.float()
    if not torch.equal(a,b): failures.append(name)
    row=dict(name=name,equal=torch.equal(a,b),relative_l2=float(d.norm()/a.float().norm()),max_abs=float(d.abs().max()),unequal=float((a!=b).float().mean()))
    results.append(row);print(json.dumps(row),flush=True)
for projection in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2'):
    a=tensor('blocks.0.'+projection+'.lora_a');b=tensor('blocks.0.'+projection+'.lora_b')
    for rows in (1023,4991):
        x=torch.randn(rows,a.shape[1]).bfloat16()
        def forward(xx):
            xp,ap,bp=[n.tensor(t) for t in (xx,a,b)]
            mid=n.empty((len(xx),128),torch.bfloat16);out=n.empty((len(xx),len(b)),torch.bfloat16)
            n.call('lora_down_bf16',mid,xp,ap,len(xx),a.shape[1])
            n.call('linear_bf16',out,mid,bp,None,len(xx),128,len(b))
            y=n.read(out,(len(xx),len(b)),torch.bfloat16);m=n.read(mid,(len(xx),128),torch.bfloat16)
            n.clear();return y,m
        y,m=forward(x);cut=(rows+1)//2
        y0,m0=forward(x[:cut]);y1,m1=forward(torch.cat((x[cut:],torch.zeros(1,a.shape[1],dtype=x.dtype))))
        compare(projection+f'-{rows}-A',m,torch.cat((m0,m1[:-1])))
        compare(projection+f'-{rows}-AB',y,torch.cat((y0,y1[:-1])))
for nq,nk in ((7,7),(898,963),(640,3100),(4860,4925),(4050,13188)):
    q=torch.randn(nq,56,128).bfloat16();k,v=[torch.randn(nk,56,128).bfloat16() for _ in range(2)]
    def forward(q,k,v):
        qp,kp,vp=[n.tensor(t) for t in (q,k,v)];out=n.empty(q.shape,torch.bfloat16)
        n.call('cross_attention_bf16',out,0,qp,0,kp,vp,0,len(q),len(k),q.shape[1])
        y=n.read(out,q.shape,torch.bfloat16);n.clear();return y
    y=forward(q,k,v)
    z=torch.cat([forward(q[:,i:i+28],k[:,i:i+28],v[:,i:i+28]) for i in (0,28)],1)
    compare(f'attention-{nq}-{nk}',y,z)
n.close()
if args.output:
    args.output.write_text(json.dumps(dict(passed=not failures,cases=results),indent=2)+'\n')
assert not failures, failures
