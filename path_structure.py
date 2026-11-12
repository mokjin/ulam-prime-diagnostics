"""Exact Ulam corner-path polynomials and local sieve signatures."""
import json
import math
from pathlib import Path
import numpy as np
from experiment import layouts

SMALL=(2,3,5,7,11)

def branch(x,y):
    # Ring k=max(|x|,|y|), maximum (2k+1)^2 at (k,-k).
    k=max(abs(x),abs(y))
    if y==-k: return 'bottom'
    if x==-k: return 'left'
    if y==k: return 'top'
    return 'right'

def value(x,y):
    k=max(abs(x),abs(y)); m=(2*k+1)**2
    return {'bottom':m-k+x,'left':m-3*k-y,'top':m-5*k-x,'right':m-7*k+y}[branch(x,y)]

def expression(x,y,sx,sy,r):
    X=x+sx*r; Y=y+sy*r
    # Select an affine expression for k, including tie convention.
    if abs(X)>=abs(Y):
        sign=1 if X>=0 else -1; u,v=sign*sx,sign*x; axis='x'
    else:
        sign=1 if Y>=0 else -1; u,v=sign*sy,sign*y; axis='y'
    edge=branch(X,Y)
    # m=4k^2+4k+1; edge adjustment is linear in k,x,y.
    ck,cx,cy={'bottom':(-1,1,0),'left':(-3,0,-1),'top':(-5,-1,0),'right':(-7,0,1)}[edge]
    a=4*u*u; b=8*u*v+(4+ck)*u+cx*sx+cy*sy
    c=4*v*v+(4+ck)*v+1+cx*x+cy*y
    return (edge,axis,sign),(a,b,c)

def segments(x,y,sx,sy,radius):
    out=[]
    for r in range(1,radius+1):
        state,coef=expression(x,y,sx,sy,r)
        if out and out[-1]['state']==state and out[-1]['coefficients']==coef:
            out[-1]['end']=r
        else: out.append(dict(start=r,end=r,state=state,coefficients=coef))
    for part in out:
        a,b,c=part['coefficients']; part['discriminant']=b*b-4*a*c
        part['roots_mod_prime']={str(ell):sum((a*t*t+b*t+c)%ell==0 for t in range(ell)) for ell in SMALL}
        vals=[a*r*r+b*r+c for r in range(part['start'],part['end']+1)]
        part['start_mod_prime']={str(ell):part['start']%ell for ell in SMALL}
        part['finite_removed_count']={str(ell):sum(n>ell and n%ell==0 for n in vals) for ell in SMALL}
        part['finite_removed_rate']={str(ell):part['finite_removed_count'][str(ell)]/len(vals) for ell in SMALL}
    return out

def signature(numbers):
    # Masks retain overlaps. A candidate equal to ell is NOT removed.
    masks=[]
    for n in numbers:
        masks.append(sum(1<<j for j,ell in enumerate(SMALL) if n>ell and n%ell==0))
    return dict(residue_counts={str(ell):np.bincount(np.array(numbers)%ell,minlength=ell).tolist() for ell in SMALL},
                occupied_residues={str(ell):len(set(n%ell for n in numbers)) for ell in SMALL},
                removed_per_prime={str(ell):sum(n>ell and n%ell==0 for n in numbers) for ell in SMALL},
                removal_mask_counts=np.bincount(masks,minlength=1<<len(SMALL)).tolist(),
                survivors=sum(m==0 and n>1 for m,n in zip(masks,numbers)))

def main():
    out=Path('results/path_structure'); out.mkdir(parents=True,exist_ok=True)
    xy=layouts(400000)['ulam']
    data=np.load('results/validation/ulam_corner.npz')
    directions=((1,1),(-1,1),(-1,-1),(1,-1))
    records=[]; checked=0; count=0; multiple=0; maximum=0
    import gzip
    output=gzip.open(out/"paths.jsonl.gz","wt",encoding="utf-8")
    for p in data['anchors']:
        x,y=map(int,xy[p])
        for j,(sx,sy) in enumerate(directions):
            parts=segments(x,y,sx,sy,20)
            ns=[value(x+sx*r,y+sy*r) for r in range(1,21)]
            for part in parts:
                a,b,c=part['coefficients']
                for r in range(part['start'],part['end']+1):
                    n=a*r*r+b*r+c
                    assert n==ns[r-1]
                    assert tuple(xy[n])==(x+sx*r,y+sy*r)
                    checked+=1
            record=dict(prime=int(p),direction=j,segments=parts,candidates=ns,
                        cumulative_signature=[signature(ns[:r]) for r in range(1,21)])
            output.write(json.dumps(record)+"\n")
            count+=1; multiple+=len(parts)>1; maximum=max(maximum,len(parts))
            if len(records)<4: records.append(record)
    output.close()
    summary=dict(paths=count,candidate_checks=checked,small_primes=SMALL,
                 multiple_segments=multiple,max_segments=maximum)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'examples.json').write_text(json.dumps(records[:4],indent=2))
    print(summary)

if __name__=='__main__': main()
