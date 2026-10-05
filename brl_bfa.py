#!/usr/bin/env python3
"""BRL bit-flip fault-attack evaluation
Injects random bit flips (Row-Hammer-like) into PTQ-quantized weights at each bitwidth,
and measures the degradation of clean accuracy and PGD robustness to study weight fault sensitivity across bitwidths.

Usage:
  python3 brl_bfa.py --ckpt ckpt/brl_rn18_at.pth --out results/bfa_at.json
"""
import argparse, json, random, torch, torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def load_base(ckpt_path):
    m = torchvision.models.resnet18(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    m.load_state_dict(ckpt, strict=False)
    return m

def make_loader(root, limit, bs=512, device="cpu"):
    tf = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    ds = datasets.CIFAR10(root=root, train=False, transform=tf, download=False)
    ds = Subset(ds, list(range(min(limit, len(ds)))))
    return DataLoader(ds, batch_size=bs, shuffle=False)

def clean_acc(m, loader, device):
    m.eval(); m.to(device); c=tot=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            c+=(m(x).argmax(1)==y).sum().item(); tot+=y.size(0)
    return c/tot*100

def pgd_robust(m, loader, eps=0.03125, iters=8, alpha=0.004, device="cuda"):
    m.eval(); m.to(device); corr=tot=0
    mean=torch.tensor([0.4914,0.4822,0.4465],device=device).view(1,3,1,1)
    std=torch.tensor([0.247,0.243,0.261],device=device).view(1,3,1,1)
    low=(-mean/std); hi=((1-mean)/std)
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        xr=x.clone().detach().requires_grad_(True)
        for _ in range(iters):
            out=m(xr); loss=torch.nn.functional.cross_entropy(out,y)
            m.zero_grad(); loss.backward(); g=xr.grad.data
            xr.data=(xr.data+alpha*g.sign()).clamp(x-eps,x+eps).clamp(low,hi)
            xr.grad.data.zero_()
        corr+=(m(xr).argmax(1)==y).sum().item(); tot+=y.size(0)
    return corr/tot*100

def q_int(w, bits):
    """per-channel symmetric quantization -> (int_repr, scale). w:(C,...) scale:(C,)"""
    shape=w.shape; wf=w.reshape(shape[0],-1)
    scale=wf.abs().max(dim=1,keepdim=True).values
    q=torch.clamp(torch.round(wf/scale* (2**(bits-1)-1)),
                  -(2**(bits-1)-1),(2**(bits-1)-1)).reshape(shape)
    return q.to(torch.int32), scale.squeeze(1)

def deq(qi, scale, bits):
    """dequant int_repr -> float weight."""
    return qi.float()/(2**(bits-1)-1) * scale.view(-1,*([1]*(qi.dim()-1)))

def flip_bits(q, n_flips, seed, bits, max_abs):
    rnd=random.Random(seed); q2=q.clone().view(-1)
    total=q2.numel()*bits
    n=min(n_flips,total)
    for idx in rnd.sample(range(q2.numel()), n):
        wv=int(q2[idx]); bitpos=rnd.randint(0,bits-1)
        mask=1<<bitpos
        wv = (wv | mask) if not (wv & mask) else (wv & ~mask)
        q2[idx]=max(-max_abs,min(max_abs,wv))
    return q2.view_as(q)

def build(sd, bits, n_flips, seed):
    nsd={}
    for k,v in sd.items():
        if "weight" in k and v.dim()>=2:
            qi,scale=q_int(v.detach().cpu(), bits)
            if n_flips: qi=flip_bits(qi, n_flips, seed, bits, 2**(bits-1)-1)
            nsd[k]=deq(qi,scale,bits)
        else:
            nsd[k]=v.clone()
    return nsd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--out", default="results/bfa.json")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--bits", default="8 6 4 3")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); random.seed(a.seed)
    base=load_base(a.ckpt); sd=base.state_dict()
    loader=make_loader(a.root, a.limit)
    bits=[int(b) for b in a.bits.split()]
    results={}
    for bbits in bits:
        # clean-only loader small; robust on smaller subset to keep time bounded
        mq=torchvision.models.resnet18(num_classes=10)
        mq.load_state_dict(build(sd,bbits,0,a.seed)); mq.to(device)
        row={"clean_base":clean_acc(mq,loader,device),
             "robust_base":pgd_robust(mq, make_loader(a.root, min(a.limit,2000)), device=device),
             "flips":{}}
        for nf in [10,100]:
            mf=torchvision.models.resnet18(num_classes=10)
            mf.load_state_dict(build(sd,bbits,nf,a.seed+bbits+nf)); mf.to(device)
            row["flips"][str(nf)]={
                "clean":clean_acc(mf,loader,device),
                "robust":pgd_robust(mf, make_loader(a.root, min(a.limit,2000)), device=device)}
        results[str(bbits)]=row
        print(f"[{bbits}-bit] base C={row['clean_base']:.2f} R={row['robust_base']:.2f} | "
              f"f10 C={row['flips']['10']['clean']:.2f} R={row['flips']['10']['robust']:.2f} | "
              f"f100 C={row['flips']['100']['clean']:.2f} R={row['flips']['100']['robust']:.2f}", flush=True)
    json.dump(results, open(a.out,"w"), indent=1); print("saved", a.out)

if __name__=="__main__":
    main()
