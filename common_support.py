"""Post-holdout descriptive ablation with frozen mod2310 strata/weights."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/prime-research-mpl')
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sieve_compare import probability,MODELS
from simulate_pairs import restricted_time

OUT=Path('results/common_support');B=500;SEED=20261122

def expectation(q):
    s=np.column_stack((np.ones(len(q)),np.cumprod(1-q,axis=1)))
    return s[:,:20].sum(1),s.mean(0)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path('results/holdout')
    files=[source/'paths.npz',source/'frozen_coefficients.json',source/'pair_simulation/mod2310.npz']
    (OUT/'manifest.json').write_text(json.dumps(dict(status='post-holdout development; not a new holdout',seed=SEED,repeats=B,inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))
    paths=np.load(files[0]);frozen=np.load(files[2]);params=json.loads(files[1].read_text())['models']
    nums=paths['numbers'];observed_y=np.minimum(paths['distance'],20)
    weights=frozen['weights'];assert weights.shape==(4,len(nums))
    unique,inverse=np.unique(nums,return_inverse=True);inverse=inverse.reshape(nums.shape)
    base=probability(unique,MODELS['mod2310'],params['mod2310']['c'])
    # Keep every candidate slot, including zero-probability ones. Never compress r.
    remove=np.zeros(len(unique),bool)
    for ell in (13,17,19):remove|=(unique>ell)&(unique%ell==0)
    deletion=base.copy();deletion[remove]=0
    variants={
        'base2310':base,
        'extra_zero_only':deletion,
        'extra_zero_and_density':probability(unique,MODELS['mod9699690'],params['mod2310']['c']),
        'stronger_frozen_c':probability(unique,MODELS['mod9699690'],params['mod9699690']['c'])}
    assert np.all(variants['extra_zero_only']<=base)
    expected={};curves={};obs={}
    for name,qu in variants.items():
        expected[name],curves[name]=expectation(qu[inverse]);obs[name]=weights@(observed_y-expected[name])
    np.testing.assert_allclose(obs['base2310'],frozen['observed'],atol=1e-12)
    sims={name:np.zeros((B,4)) for name in variants}
    rng=np.random.default_rng(SEED)
    for b in range(B):
        # Common uniforms give paired simulation comparisons across model variants.
        u=rng.random(len(unique))
        for name,qu in variants.items():
            y=restricted_time((u<qu)[inverse])
            sims[name][b]=weights@(y-expected[name])
    summary={};names=('combined','13','17','19')
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for name in variants:
        record={}
        for j,score in enumerate(names):
            vals=sims[name][:,j];K=int((abs(vals)>=abs(obs[name][j])).sum())
            record[score]=dict(observed=float(obs[name][j]),q025=float(np.quantile(vals,.025)),q975=float(np.quantile(vals,.975)),mean=float(vals.mean()),mean_mcse=float(vals.std(ddof=1)/np.sqrt(B)),tail_count=K,repeats=B,adjusted_tail_fraction=(K+1)/(B+1),tail_mcse=float(np.sqrt((K/B)*(1-K/B)/B)))
        summary[name]=record
    for j,score in enumerate(names):
        for k,name in enumerate(variants):
            rec=summary[name][score]
            axes[j].plot([k,k],[rec['q025'],rec['q975']],color='#00897b',lw=4)
            axes[j].scatter(k,rec['observed'],color='#d84315',zorder=3)
        axes[j].axhline(0,color='gray',lw=.7);axes[j].set_title(score)
        axes[j].set_xticks(range(4),list(variants),rotation=55,ha='right');axes[j].grid(axis='y',alpha=.2)
    fig.suptitle('Common mod2310 scores and strata | orange: observed | green: conditional simulation range')
    fig.tight_layout();fig.savefig(OUT/'common_support.png',dpi=160)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(OUT/'simulations.npz',weights=weights,**sims)
    lines=['# 동일 경로·점수·층에서의 확률 모델 비교','','동결 평가 결과를 확인한 뒤 설계한 후속 개발 분석이다. 기존 results/holdout 파일을 수정하지 않았다. 기준점·후보 슬롯과 반경·관측 거리·mod2310 점수·공통 지지 층·가중치는 완전히 동일하다. 새 필터 기준으로 N10이나 점수 또는 층을 다시 만들지 않았다. 제거는 q=0으로 표현하며 후보 위치를 삭제하거나 반경을 압축하지 않는다.', '', '모형 변화: (1) 기존 mod2310, (2) 13·17·19 배수의 q만 0으로 변경, (3) 같은 추가 제거에 W/φ(W) 밀도 보정까지 적용하되 c는 mod2310 값, (4) 기존에 동결한 강한 필터의 c 사용. (2)는 적정 조건부 밀도가 없는 의도적인 분해 진단이며 최종 경쟁 모델이 아니다. 어느 단계도 후보 간 의존성을 추가하지 않는다.', '', '각 모델에 고유 정수 공유 모의를 500회 수행했다. 같은 반복의 난수를 공유해 비교한다. 모형별 기대 제한 거리와 잔차를 다시 계산한다. c를 새 자료에 적합하지 않는다. 95% 범위는 모형 조건부 범위이며 신뢰구간이나 다중비교 보정된 유의성 범위가 아니다.', '', '| 모델 / 점수 | 관측 잔차 차이 | 95% 모의 범위 | K/500 |','|---|---:|---|---|']
    for name,record in summary.items():
        for score,r in record.items():lines.append(f"| {name}/{score} | {r['observed']:.4f} | [{r['q025']:.4f}, {r['q975']:.4f}] | {r['tail_count']}/500 |")
    lines+=['','이번 비교는 표본·점수 변경의 교란을 고정했지만 관측 자료를 보고 설계했고, 모형 자체가 기준 소수 선택과 실제 다후보 산술 의존성을 재현하지 않는 한계는 유지된다. 같은 관측 통계량의 잔차 기준선을 바꾼 것이므로 이탈 변화는 해당 모형 적합도에 대한 진단이다. pair-score의 인과 효과 또는 일반적 설명력을 검증한 것이 아니다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
