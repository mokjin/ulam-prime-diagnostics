"""Development-only corner-path ablation; no data beyond 400,000."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/prime-research-mpl')
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from experiment import layouts
from path_structure import value

R=20
MODELS={'density':(), 'mod30':(2,3,5), 'mod2310':(2,3,5,7,11), 'mod9699690':(2,3,5,7,11,13,17,19)}

def probability(numbers,primes,c=1.):
    n=np.asarray(numbers)
    factor=np.prod([ell/(ell-1) for ell in primes])
    q=np.zeros(n.shape,float); ok=n>1
    q[ok]=np.minimum(1,c*factor/np.log(n[ok]))
    for ell in primes: q[(n>ell)&(n%ell==0)]=0
    q[np.isin(n,primes)]=1
    return q

def survival(numbers,primes,c):
    q=probability(numbers,primes,c)
    return np.column_stack((np.ones(len(q)),np.cumprod(1-q,axis=1)))

def hazard(s):
    out=np.full(len(s)-1,np.nan)
    np.divide(s[1:],s[:-1],out=out,where=s[:-1]>0)
    return 1-out

def data(root):
    d=np.load(Path(root)/'ulam_corner.npz'); anchors=d['anchors']
    xy=layouts(400000)['ulam']; rows=[]
    # Keep the same anchor-major, direction-minor ordering as distance.ravel().
    for p in anchors:
        x,y=map(int,xy[p])
        for sx,sy in ((1,1),(-1,1),(-1,-1),(1,-1)):
            rows.append([value(x+sx*r,y+sy*r) for r in range(1,R+1)])
    return np.array(rows),d['distance'].ravel(),anchors

def main():
    out=Path('results/sieve_compare');out.mkdir(parents=True,exist_ok=True)
    train,tr,_=data('results'); test,te,anchors=data('results/validation')
    obs_train=np.array([(tr>r).mean() for r in range(R+1)])
    obs=np.array([(te>r).mean() for r in range(R+1)])
    summary={}; predictions={}; residuals={}
    np.savez_compressed(out/'observed_hazard.npz',radius=np.arange(1,R+1),at_risk=np.array([(te>=r).sum() for r in range(1,R+1)]),hazard=hazard(obs))
    for name,primes in MODELS.items():
        objective=lambda c: float(np.mean((survival(train,primes,c).mean(0)[1:]-obs_train[1:])**2))
        fit=minimize_scalar(objective,bounds=(.05,5),method='bounded',options={'xatol':1e-8}); assert fit.success
        s=survival(test,primes,float(fit.x)); mean=s.mean(0)
        assert .051<float(fit.x)<4.999, 'Calibration reached bound'
        predictions[name]=mean
        residuals[name]=np.minimum(te,R)-s[:,:R].sum(1)
        summary[name]=dict(c=float(fit.x),factor=float(np.prod([p/(p-1) for p in primes])),max_error=float(abs(mean-obs).max()),rmse=float(np.sqrt(np.mean((mean[1:]-obs[1:])**2))),mean_restricted_residual=float(residuals[name].mean()))
    # Preselected descriptive features at r<=10, among W=2310 survivors.
    # Primes 13,17,19 add information beyond the deterministic filter.
    eligible=probability(test,MODELS['mod2310'])>0
    groups=[]; scores=[]; pair_counts=[]
    for nums,ok in zip(test[:,:10],eligible[:,:10]):
        n=nums[ok]; count=len(n)*(len(n)-1)//2
        shared=sum(sum((int(n[j])-int(n[i]))%ell==0 for ell in (13,17,19)) for i in range(len(n)) for j in range(i+1,len(n)))
        score=shared/(3*count) if count else np.nan
        pair_counts.append(count);scores.append(score)
        groups.append('insufficient' if count==0 else ('none' if shared==0 else 'some'))
    descriptive={}
    for group in ('insufficient','none','some'):
        mask=np.array(groups)==group
        descriptive[group]=dict(paths=int(mask.sum()),mean_residual=float(residuals['mod2310'][mask].mean()) if mask.any() else None,mean_survivors_first10=float(eligible[mask,:10].sum(1).mean()) if mask.any() else None)
    np.savez_compressed(out/'paths.npz',anchors=anchors,distance=te,numbers=test,pair_score=scores,pair_counts=pair_counts,**{f'residual_{k}':v for k,v in residuals.items()})
    fig,ax=plt.subplots(1,2,figsize=(11,4));rr=np.arange(R+1)
    ax[0].step(rr,obs,where='post',label='Observed',lw=2)
    ax[1].plot(rr[1:],hazard(obs),label='Observed',lw=2)
    for name,s in predictions.items():
        ax[0].plot(rr,s,label=name);ax[1].plot(rr[1:],hazard(s),label=name)
    for a in ax:a.legend();a.grid(alpha=.2);a.set_xlabel('Radius r')
    ax[0].set_ylabel('Survival probability');ax[1].set_ylabel('Conditional first-discovery probability')
    fig.suptitle('Ulam corners | development range 100k–400k | coefficients fitted below 100k')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160)
    (out/'summary.json').write_text(json.dumps(dict(models=summary,pair_description=descriptive),indent=2))
    lines=['# 独立密度・体条件の要素比較（開発用）'.replace('独立密度・体条件の要素比較（開発用）','독립 밀도·체 조건의 요소 비교 (개발용)'),'', 'c는 MSE(c)=(1/20)Σ(r=1,…,20)(S_model,c(r)−S_obs(r))²를 최소화한다. scipy bounded, 범위 [0.05,5], xatol=1e-8. 연속 최적화가 반환한 해를 쓰며 근사 동률을 별도 판정하지 않는다. 해의 유일성은 주장하지 않는다. 울람 꼭짓점 경로만 비교한다. c는 모든 모델에서 동일하게 10만 이하의 평균 생존함수 오차로 추정했다. 10만 초과–40만 이하 구간은 개발용 평가이며 완전 홀드아웃이 아니다.','','| 모델 | 조건부 밀도 계수 | c | 최대 오차 | RMSE |','|---|---:|---:|---:|---:|']
    for k,v in summary.items():lines.append(f"| {k} | {v['factor']:.4f} | {v['c']:.4f} | {v['max_error']:.5f} | {v['rmse']:.5f} |")
    lines+=['','q=min(1,c·W/φ(W)/log(n)), 필터 소수 자체는 q=1, 그 소수보다 큰 배수는 q=0. W=1이면 별도 체 필터가 없다. 모든 모델은 후보 간 독립을 가정한다. mod30→mod2310 차이는 더 강한 체 정보와 그에 따른 밀도 정규화의 효과이지 의존성 모델의 성과가 아니다.','','mod 30은 기본 밀도 모델보다 개발 평가 구간을 더 잘 재현했다. mod 2310은 최대오차를 소폭 낮췄으나 RMSE는 낮추지 못했다. 따라서 두 모델의 우열을 단일 지표로 결론내리지 않는다. mod9699690은 19까지의 추가 일변량 체 조건을 반영한 독립 기준선이다. 의존성 모형은 아니다.\n\n## 후보 차이와 잔차: 탐색적 설명','','반경 10까지 mod2310을 통과한 후보쌍에 대해 차이가 13·17·19로 나누어지는지 기록했다. 점수는 공유 잔여류 관계 수 / (3 × 후보쌍 수)이며 쌍이 없으면 결측이다. 결과를 보기 전에 이 특징 정의를 코드에 고정했지만 공식 사전등록은 아니다.','','잔차는 min(R,20)−Σ(r=0,…,19)S_model(r). 각 그룹은 후보 수·숫자 크기·방향 등을 맞추지 않았으므로 평균 차이를 의존성의 인과 효과나 유의한 증거로 해석하지 않는다. 예측 회귀를 적합하지 않았다.','','| 그룹 | 경로 수 | 평균 잔차 | 반경 10 생존 후보 수 평균 |','|---|---:|---:|---:|']
    for k,v in descriptive.items():lines.append(f"| {k} | {v['paths']} | {v['mean_residual']} | {v['mean_survivors_first10']} |")
    lines+=['','위험도는 평균 생존함수의 연속 비율로 계산하며 분모 0은 결측으로 둔다. 꼬리의 관측 위험집합 수는 거리 원자료에서 계산할 수 있다. 위험집합 수는 observed_hazard.npz에도 저장했다. 현재 그림은 불확실성 구간을 제공하지 않는다. 모든 모델의 c 탐색 범위는 0.05–5로 동일하게 확장했다. 최초 범위 0.2–2에서 기본 모델이 상한에 걸린 것을 확인한 뒤 개발 단계에서 수정했으며, 홀드아웃 조정이 아니다. 다음은 후보 수·방향 등을 통제한 잔차 비교와 경계 모드 확장이다.']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
