"""Finite-candidate search benchmark; exact trial division, no label lookup."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/prime-research-mpl')
import json
import math
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiment import sieve, layouts, board

OUT=Path('results/benchmark'); OUT.mkdir(parents=True,exist_ok=True)
R=20; SAMPLE=1000; REPEATS=5

def is_prime(n):
    if n<2: return False
    if n%2==0: return n==2
    for d in range(3,math.isqrt(n)+1,2):
        if n%d==0: return False
    return True

def candidates(grid,pos,mode,sx,sy):
    x,y=pos; result=[]
    for r in range(1,R+1):
        offsets=[(r,r)] if mode=='corner' else [(r,k) for k in range(r+1)]+[(k,r) for k in range(r)]
        for dx,dy in offsets:
            yield int(grid[x+sx*dx,y+sy*dy])

def search(grid,pos,mode,sx,sy,strategy):
    seq=candidates(grid,pos,mode,sx,sy)
    if strategy in ('wheel30','density_order'):
        seq=(n for n in seq if n in (2,3,5) or math.gcd(n,30)==1)
    if strategy=='density_order':
        # c/log(n) is strictly decreasing: sorting n ascending gives the
        # same ranking without computing logs or training coefficients.
        seq=iter(sorted(seq))
    checks=0
    for n in seq:
        checks+=1
        if is_prime(n): return n,checks
    return 0,checks

prime=sieve(400000)
boards={name:board(xy,prime) for name,xy in layouts(400000).items()}
rng=np.random.default_rng(20261114)
strategies=('raw','wheel30','density_order')
summary={}; fig,axes=plt.subplots(1,2,figsize=(13,5))
for name,(grid,xy) in boards.items():
    for mode in ('corner','boundary'):
        key=f'{name}_{mode}'
        data=np.load(Path('results/validation')/f'{key}.npz')
        anchors=data['anchors']; ids=rng.choice(len(anchors)*4,SAMPLE,replace=False)
        cases=[(int(anchors[i//4]),int(i%4)) for i in ids]
        directions=((1,1),(-1,1),(-1,-1),(1,-1))
        results={}; timings={s:[] for s in strategies}
        # Rotate strategy ordering to reduce systematic warmup/order effects.
        for rep in range(REPEATS):
            for s in np.roll(strategies,rep%3):
                start=time.perf_counter()
                output=[search(grid,xy[a],mode,*directions[d],s) for a,d in cases]
                timings[s].append(time.perf_counter()-start)
                if rep==0: results[s]=np.array(output)
                else: assert np.array_equal(results[s],output)
        found=results['raw'][:,0]>0
        expected=np.array([data['distance'][np.searchsorted(anchors,a),d]<=R for a,d in cases])
        assert np.array_equal(found,expected)
        for s in strategies:
            assert np.array_equal(results[s][:,0]>0,found)
            assert all(prime[n] for n in results[s][:,0] if n)
        record={s:dict(mean_checks=float(results[s][:,1].mean()),median_total_seconds=float(np.median(timings[s])),all_total_seconds=timings[s],successes=int(found.sum())) for s in strategies}
        summary[key]=record
        np.savez_compressed(OUT/f'{key}.npz',anchors=[a for a,d in cases],directions=[d for a,d in cases],**results)
        print(key,record,flush=True)
keys=list(summary); xx=np.arange(len(keys))
for i,s in enumerate(strategies):
    axes[0].bar(xx+(i-1)*.25,[summary[k][s]['mean_checks'] for k in keys],width=.25,label=s)
    axes[1].bar(xx+(i-1)*.25,[summary[k][s]['median_total_seconds'] for k in keys],width=.25,label=s)
for ax in axes:
    ax.set_xticks(xx,keys,rotation=45,ha='right'); ax.legend(); ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Mean primality calls per search (failures included)')
axes[1].set_ylabel('Median seconds per 1,000 searches')
fig.suptitle('Same candidate region | radius <= 20 | 5 timing repeats')
fig.tight_layout(); fig.savefig(OUT/'benchmark.png',dpi=160)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# 소수 탐색 비용 실험','','10만 초과–40만 이하의 기존 공통 내부 기준 소수·방향에서 조건별 1,000쌍을 고정 난수로 추출했다. 탐색 반경은 20이다. 같은 후보 집합에서 소수 하나를 찾거나 후보를 모두 소진한다. 미발견도 비용 평균에 포함한다.','','전략: raw는 기존 경계/꼭짓점 순서. wheel30은 동일 순서에서 2·3·5의 배수를 제외한다. density_order는 같은 필터 뒤 후보를 숫자 오름차순으로 정렬한다. c/log(n)의 순위가 숫자 오름차순과 같으므로 보정계수는 탐색 순서에 영향을 주지 않는다. 이 실험은 밀도 모델에서 유도한 순서를 평가하며 새로운 학습 기반 우선순위 모델을 평가하는 것은 아니다.','','모든 전략은 같은 정확한 나눗셈 기반 소수 판정 함수를 사용한다. 체는 기준점 생성과 결과 검증에만 사용한다. 측정 시간은 후보 생성, 필터, 정렬, 판정을 포함하고, 격자 사전 생성·기준점 선택·모델 학습은 제외한다. 5회 실행의 중앙값. 정렬 전략은 후보 전체를 먼저 생성하지만 순차 전략은 첫 발견 시 생성을 멈춘다.','','| 조건 | raw 판정 수 | wheel30 판정 수 | 밀도 순서 판정 수 | wheel30 시간(s) | 밀도 순서 시간(s) |','|---|---:|---:|---:|---:|---:|']
for k,v in summary.items():
    lines.append(f"| {k} | {v['raw']['mean_checks']:.2f} | {v['wheel30']['mean_checks']:.2f} | {v['density_order']['mean_checks']:.2f} | {v['wheel30']['median_total_seconds']:.4f} | {v['density_order']['median_total_seconds']:.4f} |")
lines+=['','각 조건에서 세 전략의 성공 여부가 기존 거리 데이터와 일치하고 반환된 수가 실제 소수임을 검증했다. 순서가 달라 찾은 소수는 다를 수 있다. 최초 경계 거리나 다음으로 큰 소수를 찾는 과제와는 다르다.','','실행 시간은 이 Python 구현과 이 기기, 작은 정수 범위에 한정된다. 밀러–라빈·분할 체·대규모 정수 탐색보다 빠르다는 주장을 할 수 없다. 공간적으로 겹치는 표본이므로 독립 표본에 근거한 유의성 검정도 하지 않았다. 소수 판정 호출 수에는 gcd와 후보 생성 비용이 포함되지 않으므로 시간과 함께 해석해야 한다.']
(OUT/'REPORT.md').write_text('\n'.join(lines))
