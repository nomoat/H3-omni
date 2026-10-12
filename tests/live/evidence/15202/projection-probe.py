import json,sys,torch
sys.path.insert(0,'tests/pytorch')
from native import Native
n=Native('build-cuda/libh3_cuda.so');torch.set_num_threads(4);torch.manual_seed(47)
rows=479
for k,out in ((5376,21504),(7168,5376),(5376,28672),(14336,5376)):
 w=torch.randn(out,k).bfloat16();x=torch.randn(rows,k).bfloat16();wp=n.tensor(w)
 def forward(xx):
  xp=n.tensor(xx);yp=n.empty((len(xx),out),torch.bfloat16)
  n.call('linear_bf16',yp,xp,wp,None,len(xx),k,out)
  return n.read(yp,(len(xx),out),torch.bfloat16)
 y=forward(x)
 for parts,tile in ((2,240),(4,120),(4,256)):
  pieces=[]
  for start in range(0,rows,tile):
   xx=x[start:start+tile];valid=len(xx)
   xx=torch.cat((xx,torch.zeros(tile-valid,k,dtype=x.dtype)))
   pieces.append(forward(xx)[:valid])
  z=torch.cat(pieces)
  print(json.dumps(dict(k=k,out=out,parts=parts,tile=tile,equal=torch.equal(y,z),rel=float((y.float()-z.float()).norm()/y.float().norm()))),flush=True)
 n.clear()
for nq,nk in ((479,479),(543,543),(7,7),(898,963),(640,3100),(4860,4925),(4050,13188)):
 q=torch.randn(nq,56,128).bfloat16();k,v=[torch.randn(nk,56,128).bfloat16() for _ in range(2)]
 def forward(q,k,v):
  qp,kp,vp=[n.tensor(t) for t in (q,k,v)];yp=n.empty(q.shape,torch.bfloat16)
  n.call('cross_attention_bf16',yp,0,qp,0,kp,vp,0,len(q),len(k),q.shape[1])
  y=n.read(yp,q.shape,torch.bfloat16);n.clear();return y
 y=forward(q,k,v);z=torch.cat([forward(q[:,i:i+14],k[:,i:i+14],v[:,i:i+14]) for i in range(0,56,14)],1)
 print(json.dumps(dict(attention=[nq,nk],equal=torch.equal(y,z),rel=float((y.float()-z.float()).norm()/y.float().norm()))),flush=True)
n.close()
