"""Execute the frozen Ulam-corner replication protocol."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/prime-research-mpl')
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiment import sieve,layouts
from path_structure import value
from sieve_compare import MODELS,survival

OUT=Path('results/holdout')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=Path('paper/HOLDOUT_PROTOCOL.json').read_bytes()
    coefs=Path('results/sieve_compare/summary.json').read_bytes()
    # Save the exact specification and coefficients before generating new labels.
    (OUT/'protocol.json').write_bytes(protocol);(OUT/'frozen_coefficients.json').write_bytes(coefs)
    (OUT/'manifest.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(protocol).hexdigest(),coefficients_sha256=hashlib.sha256(coefs).hexdigest()),indent=2))
    parameters=json.loads(coefs)['models']
    prime=sieve(1000000);xy=layouts(1000000)['ulam']
    anchors=[];rows=[]
    for p in np.flatnonzero(prime[500001:900001])+500001:
        x,y=map(int,xy[p]);paths=[]
        for sx,sy in ((1,1),(-1,1),(-1,-1),(1,-1)):
            paths.append([value(x+sx*r,y+sy*r) for r in range(1,21)])
        ns=np.array(paths)
        if ns.min()<=400000 or ns.max()>1000000:continue
        anchors.append(int(p));rows.extend(paths)
    numbers=np.array(rows);anchors=np.array(anchors)
    assert numbers.min()>400000
    hits=prime[numbers];distance=np.where(hits.any(1),hits.argmax(1)+1,21)
    obs=np.array([(distance>r).mean() for r in range(21)])
    saved=dict(anchors=anchors,numbers=numbers,distance=distance)
    summary={};fig,ax=plt.subplots(figsize=(8,5));rr=np.arange(21)
    ax.step(rr,obs,where='post',label='Observed holdout',lw=2)
    for name,primes in MODELS.items():
        c=parameters[name]['c'];s=survival(numbers,primes,c);mean=s.mean(0)
        residual=np.minimum(distance,20)-s[:,:20].sum(1)
        saved['residual_'+name]=residual
        summary[name]=dict(c=c,max_error=float(abs(mean-obs).max()),rmse=float(np.sqrt(np.mean((mean[1:]-obs[1:])**2))),mean_restricted_residual=float(residual.mean()))
        ax.plot(rr,mean,label=name)
    np.savez_compressed(OUT/'paths.npz',**saved)
    meta=dict(anchors=len(anchors),paths=len(numbers),candidate_min=int(numbers.min()),candidate_max=int(numbers.max()),censored=int((distance>20).sum()))
    (OUT/'summary.json').write_text(json.dumps(dict(metadata=meta,models=summary),indent=2))
    ax.legend();ax.grid(alpha=.2);ax.set_xlabel('Radius');ax.set_ylabel('Survival probability');ax.set_title('Frozen coefficients | anchor primes 500k–900k')
    fig.tight_layout();fig.savefig(OUT/'survival.png',dpi=160)
    print(json.dumps(dict(metadata=meta,models=summary),indent=2))

if __name__=='__main__':main()
