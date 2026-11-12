"""Fixed-parameter conditional diagnostics: independent paths vs shared integers."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/prime-research-mpl')
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sieve_compare import probability,MODELS,survival
from experiment import layouts

REPEATS=500
SEED=20261120
OUT=Path('results/pair_simulation')

def restricted_time(hits):
    found=hits.any(axis=1)
    r=np.where(found,hits.argmax(axis=1)+1,21)
    return np.minimum(r,20)

def contrast_weights(n10,direction,bands,score,minimum=10):
    weights=np.zeros(len(score)); cells=[]
    for n in range(2,11):
        for d in range(4):
            for band in np.unique(bands):
                sel=(n10==n)&(direction==d)&(bands==band)
                a=np.flatnonzero(sel&(score==0));b=np.flatnonzero(sel&(score>0))
                if len(a)<minimum or len(b)<minimum:continue
                w=min(len(a),len(b));weights[a]-=w/len(a);weights[b]+=w/len(b)
                cells.append(dict(n10=n,direction=d,ring_band=int(band),none=len(a),some=len(b),weight=w))
    total=sum(c['weight'] for c in cells)
    if total:weights/=total
    return weights,cells

def main():
    global OUT,SEED
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--holdout',action='store_true')
    args=parser.parse_args()
    root=Path('results/holdout' if args.holdout else 'results/sieve_compare')
    if args.holdout:
        OUT=Path('results/holdout/pair_simulation');SEED=20261121
    OUT.mkdir(parents=True,exist_ok=True)
    d=np.load(root/'paths.npz')
    numbers=d['numbers'];anchors=d['anchors'];count=len(numbers)
    xy=layouts(1000000 if args.holdout else 400000)['ulam'];bands=np.repeat(np.abs(xy[anchors]).max(1)//10,4)
    directions=np.tile(np.arange(4),len(anchors))
    coefs=json.loads((root/('frozen_coefficients.json' if args.holdout else 'summary.json')).read_text())['models']
    unique,inverse=np.unique(numbers,return_inverse=True);inverse=inverse.reshape(numbers.shape)
    rng=np.random.default_rng(SEED)
    summary={};fig,axes=plt.subplots(2,4,figsize=(16,7))
    for row,model in enumerate(('mod2310','mod9699690')):
        c=coefs[model]['c'];primes=MODELS[model]
        q=probability(numbers,primes,c);qu=probability(unique,primes,c)
        np.testing.assert_allclose(q,qu[inverse])
        expected=survival(numbers,primes,c)[:,:20].sum(1)
        eligible=q[:,:10]>0;n10=eligible.sum(1)
        z=np.full((count,3),np.nan)
        for i,(ns,ok) in enumerate(zip(numbers[:,:10],eligible)):
            ns=ns[ok];pairs=len(ns)*(len(ns)-1)//2
            if not pairs:continue
            for j,ell in enumerate((13,17,19)):
                freqs=np.bincount(ns%ell,minlength=ell)
                z[i,j]=np.sum(freqs*(freqs-1)//2)/pairs
        scores=np.column_stack((np.mean(z,axis=1),z))
        names=('combined','13','17','19');weights=[];cells={}
        for j,name in enumerate(names):
            w,cell=contrast_weights(n10,directions,bands,scores[:,j]);weights.append(w);cells[name]=cell
        weights=np.array(weights)
        np.testing.assert_allclose(weights.sum(1),0,atol=1e-12)
        observed=weights@(np.minimum(d['distance'],20)-expected)
        sims={}
        for mode in ('independent_paths','shared_integers'):
            values=np.empty((REPEATS,4))
            for b in range(REPEATS):
                hits=(rng.random(q.shape)<q) if mode=='independent_paths' else (rng.random(len(unique))<qu)[inverse]
                values[b]=weights@(restricted_time(hits)-expected)
            sims[mode]=values
        record={}
        for j,name in enumerate(names):
            rec=dict(observed=float(observed[j]),cells=len(cells[name]),retained_paths=sum(a['none']+a['some'] for a in cells[name]),diagnostics={})
            ax=axes[row,j]
            for mode,color in (('independent_paths','#1565c0'),('shared_integers','#00897b')):
                vals=sims[mode][:,j];sd=vals.std(ddof=1);mcse=sd/np.sqrt(REPEATS)
                rec['diagnostics'][mode]=dict(mean=float(vals.mean()),sd=float(sd),mean_mcse=float(mcse),q025=float(np.quantile(vals,.025)),q975=float(np.quantile(vals,.975)),absolute_tail_fraction=float(np.mean(abs(vals)>=abs(observed[j]))),tail_count=int(np.sum(abs(vals)>=abs(observed[j]))),repeats=REPEATS,adjusted_tail_fraction=float((1+np.sum(abs(vals)>=abs(observed[j])))/(REPEATS+1)),tail_mcse=float(np.sqrt(np.mean(abs(vals)>=abs(observed[j]))*(1-np.mean(abs(vals)>=abs(observed[j])))/REPEATS)))
                ax.hist(vals,bins=25,alpha=.45,color=color,label=mode)
            ax.axvline(observed[j],color='#d84315',lw=2,label='Observed')
            ax.axvline(0,color='black',lw=.6);ax.set_title(f'{model} / {name}');ax.set_xlabel('Stratified residual difference')
            record[name]=rec
        summary[model]=dict(c=c,comparisons=record)
        np.savez_compressed(OUT/f'{model}.npz',scores=scores,n10=n10,weights=weights,observed=observed,**sims)
        (OUT/f'{model}_cells.json').write_text(json.dumps(cells,indent=2))
        print(model,json.dumps(record),flush=True)
    axes[0,0].legend(fontsize=6)
    fig.suptitle('Fixed-parameter model diagnostics | 500 simulations | observed contrasts shown in orange')
    fig.tight_layout();fig.savefig(OUT/'diagnostics.png',dpi=160)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 후보쌍 합동 관계와 독립 모델 잔차의 조건부 모의 진단','',f'반복 {REPEATS}회, 난수 시드 {SEED}. c는 기존 10만 이하 적합값으로 고정하고 재적합하지 않았다. 입력 표본은 {root}에 기록된 기준점·경로다. 정확한 N10·방향·고리 10단위 층, 그룹당 최소 10경로, min(n_none,n_some) 가중치를 유지했다. 소수별 점수마다 공통 지지 층은 다를 수 있다.', '', '통합 점수와 z13,z17,z19를 모두 저장했다. 점수는 반경 10까지 필터를 통과한 후보쌍 차이가 해당 소수로 나누어지는 비율이다. N10<2이면 결측이다. N10=2에서는 소수별 점수가 0 또는 1이므로 연속 분위수로 해석하지 않는다. 이번 비교는 0 대 양수이며 연속 점수의 단조성 분석은 수행하지 않았다.', '', '경로별 모의는 후보 출현마다 독립 Bernoulli를 생성한다. 공유 모의는 고유 후보 정수마다 한 번 생성하고 모든 경로에서 재사용한다. 둘 다 해당 W의 q(n)를 사용한다. 기준 소수의 실제 표시를 강제로 1로 고정하지 않는다. 따라서 고정된 기준 경로에서의 후보 주변분포 진단이며, 실제 소수 기준점 선택까지 재현하는 점과정 모형은 아니다.', '', '잔차 Y−E[Y], Y=min(R,20). 고정 경로·층·가중치와 올바른 주변확률 아래 두 모의의 가중 차이 기대값은 0이다. 고정 층화가 편향을 만든다고 가정하지 않는다. 모의 평균의 Monte Carlo 표준오차를 기록해 수치 점검에 사용한다.', '', '| 필터 / 점수 | 관측 차이 | 경로 독립 평균 | 경로 독립 95% 모의 범위 | 정수 공유 평균 | 정수 공유 95% 모의 범위 |','|---|---:|---:|---|---:|---|']
    for model,v in summary.items():
        for name,r in v['comparisons'].items():
            a=r['diagnostics']['independent_paths'];b=r['diagnostics']['shared_integers']
            lines.append(f"| {model}/{name} | {r['observed']:.4f} | {a['mean']:.4f} | [{a['q025']:.4f}, {a['q975']:.4f}] | {b['mean']:.4f} | [{b['q025']:.4f}, {b['q975']:.4f}] |")
    lines+=['','95% 모의 범위는 고정 파라미터·가정한 모형 아래 통계량의 범위이며 실제 소수 자료의 신뢰구간이 아니다. 저장한 absolute_tail_fraction은 이 모의 분포에서의 절대값 초과 비율이며 독립 표본 p값이나 인과 증거로 사용하지 않는다. 여러 특징을 확인한 탐색적 분석이고 훈련계수 불확실성을 포함하지 않는다. 공유 모의는 중복 정수의 일관성만 복원하며 실제 산술적 다후보 의존성을 복원하지 않는다.']
    lines += ['', '꼬리 진단량 K, B, (K+1)/(B+1), sqrt((K/B)(1−K/B)/B)를 summary.json에 저장했다. 마지막 값은 꼬리 비율의 Monte Carlo 표준오차이며 분위수 끝점의 오차가 아니다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
