import numpy as np
from skimage.draw import disk,rectangle,line

def make_ndt_phantom(size=160):
    img=np.zeros((size,size),np.float32); defects=np.zeros_like(img,bool); c=size//2
    rr,cc=disk((c,c),int(.38*size),shape=img.shape); img[rr,cc]=.72
    rr,cc=disk((c,c),int(.15*size),shape=img.shape); img[rr,cc]=0
    # boss features
    rr,cc=rectangle(start=(int(.42*size),int(.15*size)),end=(int(.58*size),int(.85*size)),shape=img.shape); img[rr,cc]=np.maximum(img[rr,cc],.82)
    for y,x,rad in [(.38,.37,.025),(.58,.63,.03),(.66,.47,.018)]:
        rr,cc=disk((int(y*size),int(x*size)),int(rad*size),shape=img.shape); img[rr,cc]=.03; defects[rr,cc]=True
    rr,cc=line(int(.30*size),int(.60*size),int(.50*size),int(.72*size));
    for dr in [-1,0,1]: img[np.clip(rr+dr,0,size-1),cc]=.02; defects[np.clip(rr+dr,0,size-1),cc]=True
    return img,defects
