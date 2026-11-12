"""Descriptive within-stratum pair/residual comparisons; no iid inference."""
from pathlib import Path
import json
import numpy as np
from experiment import layouts
from sieve_compare import probability,MODELS

OUT=Path('results/sieve_compare')

def analyze(numbers,residual,anchors,filter_name):
    eligible=probability(numbers,MODELS[filter_name])>0
    n10=eligible[:,:10].sum(1)
    score=[]
    for ns,ok in zip(numbers[:,:10],eligible[:,:10]):
        nums=ns[ok]; pairs=len(nums)*(len(nums)-1)//2
        shared=sum((int(nums[j])-int(nums[i]))%ell==0 for i in range(len(nums)) for j in range(i+1,len(nums)) for ell in (13,17,19))
        score.append(shared/(3*pairs) if pairs else np.nan)
    score=np.array(score); some=score>0; none=score==0
    xy=layouts(400000)['ulam'];rings=np.repeat(np.abs(xy[anchors]).max(1),4)
    direction=np.tile(np.arange(4),len(anchors)); ringband=rings//10
    coarse=[]
    for label in ('2','3','4','5','6+'):
        selected=(n10>=6) if label=='6+' else (n10==int(label))
        a=selected&none;b=selected&some
        coarse.append(dict(n10=label,none=int(a.sum()),some=int(b.sum()),difference=float(residual[b].mean()-residual[a].mean()) if a.any() and b.any() else None))
    # Exact N10 rather than merging 6+ in adjusted comparisons.
    cells=[]
    for n in range(2,11):
        for d in range(4):
            for band in np.unique(ringband):
                sel=(n10==n)&(direction==d)&(ringband==band)
                a=sel&none;b=sel&some
                if a.sum()<10 or b.sum()<10: continue
                cells.append(dict(n10=n,direction=d,ring_band=int(band),none=int(a.sum()),some=int(b.sum()),difference=float(residual[b].mean()-residual[a].mean())))
    weights=np.array([min(c['none'],c['some']) for c in cells])
    effect=float(np.average([c['difference'] for c in cells],weights=weights)) if len(cells) else None
    bands={}
    for band in sorted({c['ring_band'] for c in cells}):
        sub=[c for c in cells if c['ring_band']==band]
        bands[str(band)]=float(np.average([c['difference'] for c in sub],weights=[min(c['none'],c['some']) for c in sub]))
    return dict(filter=filter_name,coarse=coarse,retained_cells=len(cells),retained_paths=sum(c['none']+c['some'] for c in cells),weighted_difference=effect,ring_band_differences=bands,cells=cells)

def main():
    d=np.load(OUT/'paths.npz');results={}
    for name in ('mod2310','mod9699690'):
        results[name]=analyze(d['numbers'],d['residual_'+name],d['anchors'],name)
    (OUT/'stratified_pairs.json').write_text(json.dumps(results,indent=2))
    lines=['# 候補対'.replace('候補対','후보쌍 차이와 층화 잔차'),'', '관계는 후보쌍 차이가 13·17·19 중 하나로 나누어지는지로 정의한다. N10은 반경 10까지 해당 필터를 통과한 후보 수다. 실제 소수 발견 전까지의 생존시간을 뜻하지 않는다.', '', '먼저 N10=2,3,4,5,6+ 표를 제시한다. 보정 비교는 정확한 N10(6+로 합치지 않음), 방향, 기준점 고리 번호의 10단위 구간으로 층화한다. none/some 각각 최소 10경로인 층만 남긴다. 차이는 some−none의 평균 제한 잔차 차이이며, 통합 가중치는 두 그룹 표본수의 최솟값이다. 이 설정은 개발용 분석이며 공식 사전등록이 아니다.']
    for name,result in results.items():
        lines+=['',f'## {name}', '', '| N10 | none | some | 잔차 차이 |','|---|---:|---:|---:|']
        for c in result['coarse']:lines.append(f"| {c['n10']} | {c['none']} | {c['some']} | {c['difference']} |")
        lines+=['',f"유효 층 {result['retained_cells']}개, 경로 {result['retained_paths']}개, 가중 평균 차이 {result['weighted_difference']}.",f"고리 구간별 보정 차이: {result['ring_band_differences']}"]
    lines+=['','고리 구간은 독립적인 공간 블록이 아니다. p값·독립 표본 표준오차·인과 해석을 제공하지 않는다. 숫자 크기, 반경별 후보 수, 조각 구조, 기준 소수 잔여류의 잔여 교란이 남는다. 공통 지지가 없는 층을 제외했으므로 전체 표본 평균 효과가 아니다. 두 필터의 분석은 서로 다른 후보 집합을 사용하므로 두 분석 차이를 의존성의 추가 효과로 해석하지 않는다. 각 필터 내 비교만 탐색적으로 해석한다. 40만 초과 데이터는 사용하지 않았다.']
    (OUT/'STRATIFIED_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:{f:v[f] for f in ('retained_cells','retained_paths','weighted_difference','ring_band_differences')} for k,v in results.items()},indent=2))

if __name__=='__main__':main()
