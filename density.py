"""Candidate-density prediction calibrated only on the training range."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/prime-research-mpl')
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from experiment import sieve, layouts, board, valid_origins

R=20
OUT=Path('results/density'); OUT.mkdir(parents=True,exist_ok=True)

def candidate_hazards(grid,xy,anchors,mode):
    # Cumulative log-survival contributions: each eligible cell has
    # q(n)=c * (30/8)/log(n). No candidate primality labels are consulted.
    steps=[]
    for sx,sy in ((1,1),(-1,1),(-1,-1),(1,-1)):
        p=xy[anchors]; by_radius=[]
        for r in range(1,R+1):
            offsets=[(r,r)] if mode=='corner' else [(r,k) for k in range(r+1)]+[(k,r) for k in range(r)]
            vals=[]
            for dx,dy in offsets:
                n=grid[p[:,0]+sx*dx,p[:,1]+sy*dy]
                q=np.zeros(len(n))
                ok=(n>5)&(np.gcd(n,30)==1)
                q[ok]=3.75/np.log(n[ok])
                # Small primes are deterministic in this model.
                q[np.isin(n,[2,3,5])]=1.
                vals.append(q)
            by_radius.append(np.stack(vals,axis=1))
        steps.append(by_radius)
    return steps

def predict(steps,c):
    curves=[]
    for by_radius in steps:
        s=np.ones(len(by_radius[0])); curve=[1.]
        for q in by_radius:
            adjusted=np.where(q==1,1,np.minimum(c*q,1))
            s*=np.prod(1-adjusted,axis=1)
            curve.append(float(s.mean()))
        curves.append(curve)
    return np.mean(curves,axis=0)

def prepare(n,test):
    prime=sieve(n); bs={k:board(v,prime) for k,v in layouts(n).items()}
    valid=np.logical_and.reduce([valid_origins(g,xy,R) for g,xy in bs.values()])
    if test: valid &= np.arange(n+1)>100000
    return bs,np.flatnonzero(prime & valid)

train,aa=prepare(100000,False); test,bb=prepare(400000,True)
summary={}; fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
for col,name in enumerate(train):
    for row,mode in enumerate(('corner','boundary')):
        key=f'{name}_{mode}'
        tr=np.load(Path('results')/f'{key}.npz')
        te=np.load(Path('results/validation')/f'{key}.npz')
        assert np.array_equal(aa,tr['anchors']) and np.array_equal(bb,te['anchors'])
        observed_train=np.array([(tr['distance']>r).mean() for r in range(R+1)])
        observed_test=te['survival']
        hs=candidate_hazards(*train[name],aa,mode)
        objective=lambda c: float(np.mean((predict(hs,c)[1:]-observed_train[1:])**2))
        fitted=minimize_scalar(objective,bounds=(.2,2),method='bounded')
        assert fitted.success
        test_steps=candidate_hazards(*test[name],bb,mode)
        prediction=predict(test_steps,float(fitted.x))
        uncalibrated=predict(test_steps,1.)
        old=te['training_prediction']
        errors={label:dict(max_error=float(abs(curve-observed_test).max()),rmse=float(np.sqrt(np.mean((curve[1:]-observed_test[1:])**2)))) for label,curve in [('old_weibull',old),('density',prediction),('density_uncalibrated',uncalibrated)]}
        summary[key]=dict(calibration=float(fitted.x),errors=errors)
        np.savez_compressed(OUT/f'{key}.npz',observed=observed_test,prediction=prediction,uncalibrated=uncalibrated,old=old)
        ax=axes[row,col]; rr=np.arange(R+1)
        ax.step(rr,observed_test,where='post',label='Observed new range',lw=2)
        ax.plot(rr,old,'--',label='Old Weibull',color='#ef6c00')
        ax.plot(rr,prediction,label='Density, calibrated on old range',color='#00897b')
        ax.plot(rr,uncalibrated,':',label='Density, no calibration',color='#9c27b0')
        ax.set_title(f'{name} / {mode}'); ax.set_xlabel('Radius r'); ax.set_ylabel('P(R > r)'); ax.grid(alpha=.2)
        print(key,errors,flush=True)
axes[0,0].legend(fontsize=6)
fig.suptitle('Density-aware prediction | calibration <= 100k | test anchor range 100k–400k')
fig.tight_layout(); fig.savefig(OUT/'density.png',dpi=160)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# 숫자 크기에 따른 밀도 보정 실험','','각 후보 정수 n에 대해 2·3·5의 배수(그 소수 자체 제외)는 확률 0, 그 외에는 q(n)=min(1, c × 3.75/log(n))으로 둔다. 3.75=30/8은 mod 30에서 소수 후보가 될 수 있는 잔여류의 비율을 보정한다. 이는 독립 후보를 가정한 탐색용 근사 모델이다.','','각 기준점·방향에서 실제 검사하는 후보들의 (1-q(n))을 곱해 생존확률을 구하고 평균한다. 따라서 검사 칸의 숫자와 산술 조건을 반영한다. 후보의 실제 소수 여부는 예측 계산에 쓰지 않는다.','','c는 배열·탐색 방식별로 10만 이하의 경험적 생존함수와의 평균제곱오차를 최소화해 결정한다. 새 구간에서는 재적합하지 않는다. c=1인 무보정 모델도 함께 저장했다. 모든 오차는 확률 단위이며 0.01은 1%포인트다.','','| 조건 | c | 이전 최대오차 | 밀도 모델 최대오차 | 이전 RMSE | 밀도 RMSE |','|---|---:|---:|---:|---:|---:|']
for k,v in summary.items():
    a=v['errors']['old_weibull']; b=v['errors']['density']
    lines.append(f"| {k} | {v['calibration']:.3f} | {a['max_error']:.4f} | {b['max_error']:.4f} | {a['rmse']:.4f} | {b['rmse']:.4f} |")
lines+=['','이 비교는 밀도·합동조건·후보 기하를 함께 바꾼 모델 비교다. 개선을 밀도 효과 하나에만 귀속할 수 없다. 기준 소수가 주어졌다는 조건으로 예측하며, 일반 정수에서 기준 소수를 찾는 비용은 포함하지 않는다. 서로 겹치는 탐색 영역, 경계를 넘는 후보, 표본 구성 차이 등 앞 실험의 한계도 유지된다. 새로운 소수 법칙 또는 탐색 속도 향상의 증명은 아니다.','','다음 검증에서는 겹치지 않는 공간 블록과 추가 숫자 구간을 사용하고, 실제 소수 판정 횟수와 후보 생성 비용을 측정해야 한다.']
(OUT/'REPORT.md').write_text('\n'.join(lines))
