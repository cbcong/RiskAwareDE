"""Independent geometry/AVaR checks plus two small actual planning runs."""
from pathlib import Path
import json, time
import numpy as np
from scipy.optimize import linprog
from adapter import continuous_losses, empirical_avar, NORMALS, plan
BASE=Path(__file__).resolve().parent
rng=np.random.default_rng(5015)
loss_error=[];gradient_error=[];collision_agreement=[]
def slab_hit(a,b,lo,hi):
    d=b-a; enter=0.;leave=1.
    for k in range(3):
        if abs(d[k])<1e-12:
            if a[k]<lo[k] or a[k]>hi[k]:return False
        else:
            t1,t2=sorted(((lo[k]-a[k])/d[k],(hi[k]-a[k])/d[k]))
            enter=max(enter,t1);leave=min(leave,t2)
    return enter<=leave
for i in range(120):
    pts=rng.uniform(-4,4,(5,3));c=rng.normal(size=3);h=rng.uniform(.1,2,3)
    val,grad=continuous_losses(pts,c[None,None,:],h[None,:],True)
    for s in range(4):
        a=NORMALS@(pts[s]-c)-np.r_[h,h]
        b=NORMALS@(pts[s+1]-pts[s])
        lp=linprog([0,1],A_ub=np.c_[b,-np.ones(6)],b_ub=-a,bounds=[(0,1),(None,None)],method='highs')
        assert lp.success
        loss_error.append(abs(val[0,0,s]+lp.fun))
        collision_agreement.append((val[0,0,s]>=0)==slab_hit(pts[s],pts[s+1],c-h,c+h))
    fd=np.zeros_like(grad)
    for k in range(9):
        pp=pts.copy();pm=pts.copy();p,j=divmod(k,3)
        pp[p+1,j]+=1e-5;pm[p+1,j]-=1e-5
        fd[...,k]=(continuous_losses(pp,c[None,None,:],h[None,:])-continuous_losses(pm,c[None,None,:],h[None,:]))/2e-5
    gradient_error.append(float(np.max(np.abs(fd-grad))))
z=rng.normal(size=32);alpha=.1
# Independently solve canonical sample AVaR epigraph LP.
q=np.r_[1.,np.ones(32)/(32*alpha)]
A=np.c_[-np.ones(32),-np.eye(32)]
lp=linprog(q,A_ub=A,b_ub=-z,bounds=[(None,None)]+[(0,None)]*32,method='highs')
avar_error=abs(lp.fun-empirical_avar(z,alpha))
cases=[]
for name,boxes in [('empty',[]),('single_box',[[[8,6,0],[12,14,8]]])]:
    r=plan([1,1,1],[19,19,9],boxes,[[0,0,0],[20,20,10]],seed=4,budget_seconds=.7,sigma=.6,samples=24)
    np.savez_compressed(BASE/f'adapter_smoke_{name}.npz',points=r.pop('points'),controls=r.pop('controls'),segment_times=r.pop('segment_times'))
    r['case']=name;cases.append(r)
report={'continuous_loss_vs_independent_LP_cases':len(loss_error),'maximum_loss_error':max(loss_error),
        'collision_vs_independent_slab_all_agree':all(collision_agreement),'analytic_gradient_vs_finite_difference_max_error':max(gradient_error),
        'empirical_AVaR_vs_independent_LP_error':avar_error,'planning_smokes':cases,
        'passed':bool(max(loss_error)<1e-8 and all(collision_agreement) and max(gradient_error)<1e-6 and avar_error<1e-9)}
(BASE/'adapter_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='planning_smokes'},indent=2))
for r in cases:print(r['case'],r['status'],r['elapsed_seconds'],r['sample_avar'],r['iterations'])
assert report['passed']
