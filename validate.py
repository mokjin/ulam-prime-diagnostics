"""Out-of-range validation with repeated, arithmetic-aware controls."""
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/prime-research-mpl")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiment import sieve, layouts, board, valid_origins, distances, fit

OUT=Path('results/validation'); OUT.mkdir(parents=True,exist_ok=True)
N=400000; R=20; REPEATS=20
prime=sieve(N); maps=layouts(N)
boards={k:board(v,prime) for k,v in maps.items()}
eligible=np.logical_and.reduce([valid_origins(g,xy,R) for g,xy in boards.values()])
number=np.arange(N+1)
eligible &= number>100000
anchors=np.flatnonzero(prime & eligible)
training=json.loads(Path('results/summary.json').read_text())
radii=np.arange(R+1)
rng=np.random.default_rng(20261113)
controls={}
# Permutations within number blocks and residue classes preserve the exact
# number of labels in each arithmetic stratum. Each control uses its own anchors.
for modulus in (2,30):
    runs=[]
    for rep in range(REPEATS):
        labels=prime.copy()
        for low in range(1,N+1,5000):
            ids=np.arange(low,min(low+5000,N+1)); ids=ids[ids>5]
            for residue in range(modulus):
                group=ids[ids%modulus==residue]
                labels[group]=rng.permutation(labels[group])
                assert labels[group].sum() == prime[group].sum()
        runs.append(labels)
    controls[modulus]=runs

summary={}; fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
for col,(name,(grid,xy)) in enumerate(boards.items()):
    for row,mode in enumerate(('corner','boundary')):
        key=f'{name}_{mode}'
        raw=distances(grid,xy,anchors,prime,mode,R)
        survival=np.array([(raw>r).mean() for r in radii])
        old=training[key]['fits']['weibull']
        predicted=np.exp(-old['alpha']*radii**old['beta'])
        new=fit(raw.ravel(),R,True)
        ax=axes[row,col]
        ax.step(radii,survival,where='post',color='#1565c0',lw=2,label='Primes: 100k < anchor <= 400k')
        ax.plot(radii,predicted,'--',color='#ef6c00',label='Model fitted below 100k')
        record=dict(anchors=len(anchors),test_beta=new['beta'],test_alpha=new['alpha'],fit_converged=new['converged'],training_model_max_survival_error=float(abs(survival-predicted).max()),censored=int((raw>R).sum()),directions={})
        for j,d in enumerate(('++','-+','--','+-')):
            record['directions'][d]=dict(restricted_mean=float(np.minimum(raw[:,j],R).mean()),censored=int((raw[:,j]>R).sum()))
        saved=dict(anchors=anchors,distance=raw,survival=survival,training_prediction=predicted)
        for modulus,color in ((2,'#999999'),(30,'#00897b')):
            curves=[]
            for labels in controls[modulus]:
                aa=np.flatnonzero(labels & eligible)
                dd=distances(grid,xy,aa,labels,mode,R)
                curves.append([(dd>r).mean() for r in radii])
            curves=np.array(curves); mean=curves.mean(axis=0)
            ax.plot(radii,mean,color=color,label=f'Mod {modulus} control mean')
            ax.fill_between(radii,curves.min(axis=0),curves.max(axis=0),color=color,alpha=.2)
            record[f'mod_{modulus}_max_mean_difference']=float(abs(survival-mean).max())
            saved[f'mod_{modulus}_survival']=curves
        summary[key]=record
        np.savez_compressed(OUT/f'{key}.npz',**saved)
        ax.set_title(f'{name} / {mode}'); ax.set_xlabel('Radius r'); ax.set_ylabel('P(R > r)'); ax.set_ylim(0,1); ax.grid(alpha=.2)
        print(key,record['training_model_max_survival_error'],flush=True)
axes[0,0].legend(fontsize=6)
fig.suptitle('New anchor range validation | shaded areas: min–max of 20 control realizations')
fig.tight_layout(); fig.savefig(OUT/'validation.png',dpi=160)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# 후속 실험: 더 큰 기준 소수와 산술 대조군','',f'기준 소수: 100,000 < p ≤ 400,000. 공통 내부 기준점 {len(anchors):,}개. 탐색 거리 20. 일반 격자 너비는 기존 315·316·317 유지.','', '| 배열 / 탐색 | 새 구간 beta | 이전 모델 최대 오차 | mod 2 대조 평균과 차이 | mod 30 대조 평균과 차이 |','|---|---:|---:|---:|---:|']
for k,v in summary.items():
    lines.append(f"| {k} | {v['test_beta']:.3f} | {v['training_model_max_survival_error']:.3f} | {v['mod_2_max_mean_difference']:.3f} | {v['mod_30_max_mean_difference']:.3f} |")
lines += ['', '오차는 r=0,…,20에서 생존확률 차이의 최댓값이다. 새 구간 beta는 새 데이터에 다시 적합한 기술통계이며, 이전 모델 오차는 첫 실험의 계수를 고정하여 계산했다.', '', '대조군은 5,000개 정수 구간마다 mod 2 또는 mod 30 잔여류 안에서 소수 표시를 순열로 섞었다. 2·3·5는 고정했다. mod 30은 2·3·5의 배수 구조와 잔여류별 소수 개수를 보존한다. 각 조건 20회, 시드 20261113. 각 대조군은 자체 기준점을 사용한다.', '', '그림의 음영은 20회 실현의 최솟값–최댓값이며 신뢰구간이 아니다. 차이가 작아지면 그 조건에서 산술 구조를 보존한 대조군이 관측 분포를 더 잘 재현한다는 뜻이다. 인과적 설명이나 새로운 법칙의 증명은 아니다.', '', '이 검증은 기준 소수 범위를 분리한다. 탐색 후보가 100,000 경계를 넘나들 수 있으므로 완전한 공간 독립 검증은 아니다. 기준점과 방향도 겹치는 탐색 영역을 가지므로 독립 관측으로 취급하지 않는다. 전체 자연수 범위를 확장하면서 공통 내부 표본의 구성도 달라진다.', '', '다음 단계: 수 크기에 따른 밀도 보정 모델, 겹침을 줄인 공간 블록 검증, 후보 검사 횟수 기준 탐색 효율 평가.']
(OUT/'REPORT.md').write_text('\n'.join(lines))
