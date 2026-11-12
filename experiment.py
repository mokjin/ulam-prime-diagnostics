"""Corner-anchored prime search; run: python3 experiment.py."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/prime-research-mpl')
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N, LIMIT = 100_000, 20
OUT = Path('results')

def sieve(n):
    a = np.ones(n+1, dtype=bool); a[:2] = False
    for p in range(2, int(n**0.5)+1):
        if a[p]: a[p*p::p] = False
    return a

def layouts(n):
    xy = np.zeros((n+1, 2), dtype=int)
    x = y = 0; k = 1; length = 1
    while k < n:
        for dx, dy in ((1,0),(0,1),(-1,0),(0,-1)):
            for _ in range(length):
                if k == n: break
                x += dx; y += dy; k += 1; xy[k] = x,y
            if dy: length += 1
    result = {'ulam': xy}
    for w in (315,316,317):
        a = np.arange(n+1)-1
        result[f'grid_{w}'] = np.column_stack((a%w, a//w))
    return result

def board(coords, labels):
    c = coords[1:]; lo = c.min(axis=0)
    shifted = coords-lo
    grid = np.zeros(tuple(c.max(axis=0)-lo+1), dtype=int)
    grid[shifted[1:,0], shifted[1:,1]] = np.arange(1,len(coords))
    return grid, shifted

def valid_origins(grid, xy, radius):
    # Require the entire centered box: all four anchored squares are observable.
    valid = grid > 0
    sat = np.pad(valid.astype(int), ((1,0),(1,0))).cumsum(0).cumsum(1)
    x,y = xy.T; a=x-radius; b=y-radius; c=x+radius+1; d=y+radius+1
    ok=(a>=0)&(b>=0)&(c<=grid.shape[0])&(d<=grid.shape[1])
    ids=np.flatnonzero(ok)
    ok[ids] = (sat[c[ids],d[ids]]-sat[a[ids],d[ids]]-sat[c[ids],b[ids]]+sat[a[ids],b[ids]]) == (2*radius+1)**2
    ok[0]=False
    return ok

def distances(grid, xy, anchors, labels, mode, radius):
    values=[]
    for sx,sy in ((1,1),(-1,1),(-1,-1),(1,-1)):
        pos=xy[anchors]; rfirst=np.full(len(anchors),radius+1)
        for r in range(1,radius+1):
            ids=np.flatnonzero(rfirst>radius)
            if not len(ids): break
            p=pos[ids]; hit=np.zeros(len(ids),bool)
            offsets=[(r,r)] if mode=='corner' else [(r,k) for k in range(r+1)]+[(k,r) for k in range(r)]
            for dx,dy in offsets:
                hit |= labels[grid[p[:,0]+sx*dx,p[:,1]+sy*dy]]
            rfirst[ids[hit]]=r
        values.append(rfirst)
    return np.column_stack(values)

def fit(data, radius, weibull):
    r=data[data<=radius].astype(float); censored=np.sum(data>radius)
    def objective(theta):
        alpha=np.exp(theta[0]); beta=np.exp(theta[1]) if weibull else 1.
        left=alpha*(r-1)**beta; delta=alpha*r**beta-left
        return float(np.sum(left-np.log(-np.expm1(-delta)))+censored*alpha*radius**beta)
    opt=minimize(objective, [-2.,0.] if weibull else [-2.], bounds=[(-15,5)]*1+([(-3,2)] if weibull else []))
    return dict(alpha=float(np.exp(opt.x[0])),beta=float(np.exp(opt.x[1])) if weibull else 1.,nll=float(opt.fun),converged=bool(opt.success))

def main():
    OUT.mkdir(exist_ok=True)
    prime=sieve(N); maps=layouts(N)
    boards={name:board(xy,prime) for name,xy in maps.items()}
    eligible=np.logical_and.reduce([valid_origins(g,xy,LIMIT) for g,xy in boards.values()])
    anchors=np.flatnonzero(prime & eligible)
    rng=np.random.default_rng(20261112)
    random=prime.copy()
    for low in range(1,N+1,5000):
        ids=np.arange(low,min(low+5000,N+1)); ids=ids[(ids%2==1)&(ids>2)]
        random[ids]=rng.permutation(random[ids])
    random_anchors=np.flatnonzero(random & eligible)
    summary={}; fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    rr=np.arange(LIMIT+1)
    for col,(name,(grid,xy)) in enumerate(boards.items()):
        for row,mode in enumerate(('corner','boundary')):
            raw=distances(grid,xy,anchors,prime,mode,LIMIT)
            null=distances(grid,xy,random_anchors,random,mode,LIMIT)
            data=raw.ravel(); baseline=null.ravel()
            key=f'{name}_{mode}'
            np.savez_compressed(OUT/f'{key}.npz',anchors=anchors,distance=raw,random_anchors=random_anchors,random_distance=null)
            fits={m:fit(data,LIMIT,m=='weibull') for m in ('exponential','weibull')}
            summary[key]=dict(anchors=len(anchors),observations=len(data),censored=int(np.sum(data>LIMIT)),restricted_mean=float(np.minimum(data,LIMIT).mean()),fits=fits)
            ax=axes[row,col]
            ax.step(rr,[(data>r).mean() for r in rr],where='post',label='Primes',color='#1565c0',lw=2)
            ax.step(rr,[(baseline>r).mean() for r in rr],where='post',label='Shuffled odd labels',color='#888888')
            for m,f in fits.items(): ax.plot(rr,np.exp(-f['alpha']*rr**f['beta']), '--',label=m)
            ax.set_title(f'{name} / {mode}'); ax.set_ylim(0,1); ax.grid(alpha=.2)
            ax.set_xlabel('Radius r'); ax.set_ylabel('P(R > r)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'First discovery survival | n <= {N:,} | common interior anchors | radius <= {LIMIT}')
    fig.tight_layout(); fig.savefig(OUT/'survival.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,name in zip(axes,('ulam','grid_316')):
        ids=np.flatnonzero(prime); coords=maps[name][ids]
        ax.scatter(*coords.T,s=.7,c='#1565c0'); ax.set_aspect('equal'); ax.set_title(name)
    fig.tight_layout(); fig.savefig(OUT/'prime_maps.png',dpi=160); plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
