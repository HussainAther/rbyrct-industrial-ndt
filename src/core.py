import numpy as np
from skimage.transform import radon,iradon
from skimage.metrics import structural_similarity

def angles(n=180): return np.linspace(0,180,n,endpoint=False)
def acquire(img,a): return radon(img,theta=a,circle=True)
def reconstruct(sino,a,size): return np.clip(np.nan_to_num(iradon(sino,theta=a,circle=True,filter_name='ramp',output_size=size)),0,1)
def uniform_idx(nall,n): return np.unique(np.linspace(0,nall-1,n,dtype=int))
def random_idx(nall,n,rng): return np.sort(rng.choice(nall,n,replace=False))
def _dist(a,b):
    d=np.abs(b-a); return np.minimum(d,180-d)
def adaptive_idx(sino,a,n,size,seed=6):
    selected=np.unique(np.linspace(0,len(a)-1,min(seed,n),dtype=int)).tolist()
    if len(selected)>=n: return np.array(sorted(selected))
    sel=np.array(sorted(selected)); rec=reconstruct(sino[:,sel],a[sel],size)
    gy,gx=np.gradient(rec); edge=np.hypot(gx,gy)
    candidates=np.array([i for i in range(len(a)) if i not in selected])
    p=radon(edge,theta=a[candidates],circle=True)
    info=np.var(p,axis=0); info=(info-info.min())/(np.ptp(info)+1e-9)
    priority={int(i):float(v) for i,v in zip(candidates,info)}
    while len(selected)<n:
        cand=np.array([i for i in range(len(a)) if i not in selected]); sa=a[np.array(selected)]
        div=np.array([np.min(_dist(a[i],sa)) for i in cand]); div/=max(div.max(),1e-9)
        inf=np.array([priority.get(int(i),0.0) for i in cand])
        selected.append(int(cand[np.argmax(.25*inf+1.0*div)]))
    return np.array(sorted(selected))
def metrics(ref,rec,defects):
    mse=float(np.mean((ref-rec)**2)); s=float(structural_similarity(ref,rec,data_range=1.0))
    yy,xx=np.indices(ref.shape); c=(ref.shape[0]-1)/2.0
    r=np.hypot(yy-c,xx-c)
    # Known inspection envelope of the manufactured annular part.
    inspect=(r<=ref.shape[0]*.39)&(r>=ref.shape[0]*.14)
    pred=(rec<.25)&inspect
    tp=np.logical_and(pred,defects).sum(); recall=float(tp/max(defects.sum(),1)); union=np.logical_or(pred,defects).sum(); iou=float(tp/union) if union else 1.0
    return mse,s,recall,iou
