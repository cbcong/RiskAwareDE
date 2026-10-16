"""SAA-AVaR-3D: documented geometric adaptation of Lew et al. (RA-L).

Not the original drone dynamics experiment. This retains sample-average AVaR
epigraph constraints and sequential convex programming with OSQP, and uses a
first-order kinematic path, exact continuous segment/box losses, and a control-
energy objective. Three free waypoints are the default, not a paper setting.
Only NumPy, SciPy and OSQP are required; JAX is needed only by native_smoke.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
import osqp

METHOD = 'SAA-AVaR-3D (geometric adaptation)'
NORMALS = np.r_[np.eye(3), -np.eye(3)]
PAIR_I, PAIR_J = np.triu_indices(6, 1)


def _boxes_array(boxes):
    """Accept [N,2,3] min/max, [N,6] xmin/ymin/zmin/xmax/ymax/zmax.

    Dict lists may use min/max, lower/upper or center/half_size.
    """
    if len(boxes) == 0:
        return np.empty((0,2,3))
    if isinstance(boxes[0], dict):
        arr = []
        for b in boxes:
            if 'min' in b: lo, hi = b['min'], b['max']
            elif 'lower' in b: lo, hi = b['lower'], b['upper']
            elif 'center' in b:
                c = np.asarray(b['center'],float)
                h = np.asarray(b.get('half_size',b.get('half_extents')),float)
                lo, hi = c-h, c+h
            else: raise ValueError('Unsupported box dict; provide explicit min/max')
            arr.append([lo,hi])
        boxes = arr
    arr = np.asarray(boxes,float)
    if arr.ndim==2 and arr.shape[1]==6: arr=arr.reshape(-1,2,3)
    if arr.ndim!=3 or arr.shape[1:]!=(2,3) or np.any(arr[:,1]<arr[:,0]):
        raise ValueError('boxes must be [N,2,3] with lower then upper corners')
    return arr


def continuous_losses(points, centers, half_sizes, gradients=False):
    """Exact whole-segment negative L-infinity signed box distance.

    For each segment and box, min_t max_f(a_f+b_f*t) is attained at
    endpoints or intersections of two of six affine box faces. Its negative
    is positive iff the continuous segment penetrates the box interior.
    centers: [samples,boxes,3]. Returns [samples,boxes,segments], and optional
    analytic subgradient w.r.t. all free 3D waypoints.
    """
    points = np.asarray(points,float)
    center = np.asarray(centers,float)
    half = np.asarray(half_sizes,float)
    starts = points[:-1]
    delta = np.diff(points,axis=0)
    a = np.einsum('mbsk,fk->mbsf', starts[None,None,:,:]-center[:,:,None,:], NORMALS)
    a -= np.concatenate([half,half],axis=-1)[None,:,None,:]
    slopes = delta @ NORMALS.T
    den = slopes[:,PAIR_I]-slopes[:,PAIR_J]
    num = a[...,PAIR_J]-a[...,PAIR_I]
    cand = np.divide(num,den[None,None,:,:],out=np.zeros_like(num),where=np.abs(den[None,None,:,:])>1e-13)
    valid = (cand>=0)&(cand<=1)&(np.abs(den[None,None,:,:])>1e-13)
    cand = np.clip(cand,0,1)
    times = np.concatenate([np.zeros(a.shape[:-1]+(1,)),np.ones(a.shape[:-1]+(1,)),cand],axis=-1)
    vals = a[...,None,:]+times[...,None]*slopes[None,None,:,None,:]
    worst = np.max(vals,axis=-1)
    worst[...,2:] = np.where(valid,worst[...,2:],np.inf)
    best = np.argmin(worst,axis=-1)
    losses = -np.take_along_axis(worst,best[...,None],axis=-1)[...,0]
    if not gradients: return losses
    ts = np.take_along_axis(times,best[...,None],axis=-1)[...,0]
    actvals = a+ts[...,None]*slopes[None,None,:,:]
    active = np.argmax(actvals,axis=-1)
    normals = NORMALS[active].copy()
    # Interior minimizers use a convex combination of two active face normals
    # whose directional derivative along the segment is zero.
    interior = (best>=2)&(ts>1e-10)&(ts<1-1e-10)
    pair_index = np.clip(best-2,0,len(PAIR_I)-1)
    fi,fj = PAIR_I[pair_index],PAIR_J[pair_index]
    ss = np.broadcast_to(slopes[None,None,:,:],a.shape)
    bi = np.take_along_axis(ss,fi[...,None],axis=-1)[...,0]
    bj = np.take_along_axis(ss,fj[...,None],axis=-1)[...,0]
    w = np.divide(-bj,bi-bj,out=np.zeros_like(bi),where=np.abs(bi-bj)>1e-13)
    combo = w[...,None]*NORMALS[fi]+(1-w[...,None])*NORMALS[fj]
    # When a non-active pair gives the same min by numerical coincidence,
    # use all active opposing-slope faces for a valid subgradient.
    maximum = np.max(actvals,axis=-1)
    fi_active = np.abs(np.take_along_axis(actvals,fi[...,None],axis=-1)[...,0]-maximum)<1e-8
    fj_active = np.abs(np.take_along_axis(actvals,fj[...,None],axis=-1)[...,0]-maximum)<1e-8
    use = interior&fi_active&fj_active&(w>=-1e-8)&(w<=1+1e-8)
    normals[use] = combo[use]
    grad = np.zeros(losses.shape+(3*(len(points)-2),))
    for s in range(len(points)-1):
        if s>0: grad[:,:,s,3*(s-1):3*s] = -(1-ts[:,:,s,None])*normals[:,:,s]
        if s<len(points)-2: grad[:,:,s,3*s:3*(s+1)] = -ts[:,:,s,None]*normals[:,:,s]
    return losses,grad


def empirical_avar(losses, alpha):
    z = np.asarray(losses,float).reshape(-1)
    if len(z)==0: return float('-inf')
    # Convex piecewise-linear minimizer is one of the sample losses.
    ts = np.unique(z)
    return float(np.min(ts+np.maximum(z[:,None]-ts[None,:],0).mean(axis=0)/alpha))


def plan(start, goal, boxes, bounds, seed=0, budget_seconds=1., sigma=.6,
         *, alpha=.1, samples=32, free_waypoints=3, max_iterations=80,
         initial_points=None, trust_radius=3., slack_penalty=10000., initialization_trials=32):
    """Plan in seconds; returns real piecewise-linear kinematic trajectory.

    All setup and solver calls are charged. A single in-flight iteration can
    overrun the requested deadline: actual/overrun seconds are always returned.
    A finite output is not necessarily AVaR-feasible; inspect risk_feasible and
    nominal_collision. No infeasible/timeout runs are hidden.
    """
    started = time.perf_counter()
    start,goal=np.asarray(start,float),np.asarray(goal,float)
    bb=_boxes_array(boxes)
    limits=np.asarray(bounds,float)
    if limits.shape==(3,2): limits=limits.T
    if limits.shape!=(2,3): raise ValueError('bounds must be [2,3] lower/upper or [3,2]')
    if not(0<alpha<1) or samples<1 or free_waypoints<1 or budget_seconds<=0:
        raise ValueError('positive sample/waypoint counts and budget; alpha in (0,1)')
    rng=np.random.default_rng(seed)
    center=(bb[:,0]+bb[:,1])/2
    half=(bb[:,1]-bb[:,0])/2
    shifts=rng.normal(0,sigma,(samples,len(bb),3))
    centers=center[None,:,:]+shifts
    count=free_waypoints+2
    if initial_points is None:
        pts=np.linspace(start,goal,count)
        # Tiny deterministic-with-seed perturbation prevents symmetric face ties;
        # this initialization change is explicit in the adaptation record.
        pts[1:-1]+=rng.normal(0,.01,(free_waypoints,3))
    else:
        pts=np.asarray(initial_points,float).copy()
        if pts.shape!=(count,3): raise ValueError('initial_points shape mismatch')
        pts[0],pts[-1]=start,goal
    pts[1:-1]=np.clip(pts[1:-1],limits[0],limits[1])
    d=3*free_waypoints; n=d+samples+2; yi=slice(d,d+samples); ti=n-2; si=n-1
    # T=1, controls are segment displacement / dt; integral ||u||^2 dt.
    segs=count-1
    D=np.zeros((segs*3,d)); offset=np.zeros((segs,3))
    for s in range(segs):
        if s>0: D[3*s:3*s+3,3*(s-1):3*s]-=np.eye(3)
        else: offset[s]-=start
        if s<free_waypoints: D[3*s:3*s+3,3*s:3*s+3]+=np.eye(3)
        else: offset[s]+=goal
    off=offset.reshape(-1)
    H=2*segs*(D.T@D); q=np.zeros(n);q[:d]=2*segs*D.T@off;q[si]=slack_penalty
    P=sparse.block_diag((sparse.csc_matrix(H),sparse.csc_matrix((samples+1,samples+1)),sparse.csc_matrix([[slack_penalty]])),format='csc')
    energy=lambda p: float(segs*np.square(np.diff(p,axis=0)).sum())
    def evalp(p):
        if len(bb)==0:return -1e6,False
        g=continuous_losses(p,centers,half)
        ar=empirical_avar(g.max(axis=(1,2)),alpha)
        nominal=bool(continuous_losses(p,center[None,:,:],half).max()>=-1e-10)
        return ar,nominal
    ar,nominal=evalp(pts)
    initialization_evaluations=1
    init_started=time.perf_counter()
    # A nonsmooth box crossing has genuine flat penetration-loss regions. Use a
    # bounded, recorded random-arch initializer before SCP rather than presenting
    # straight-line stagnation as a faithful successful external baseline. This
    # geometric front end is added by this adaptation and is charged to budget.
    # Candidate ranking uses ONLY the same planning samples, never external data.
    if initial_points is None and len(bb):
        base=np.linspace(start,goal,count)
        phase=np.sin(np.linspace(0,np.pi,count))[:,None]
        scale=.45*(limits[1]-limits[0])
        for j in range(int(initialization_trials)):
            if time.perf_counter()-started>=.25*budget_seconds: break
            if j<6:
                direction=np.zeros(3);direction[j//2]=1. if j%2 else -1.
            else: direction=rng.uniform(-1,1,3)
            candidate=base+phase*(scale*direction)
            if j>=12:
                candidate[1:-1]+=rng.normal(0,.5,(free_waypoints,3))
            candidate[1:-1]=np.clip(candidate[1:-1],limits[0],limits[1])
            car,cnom=evalp(candidate);initialization_evaluations+=1
            current=(int(nominal),int(ar>1e-5),max(ar,0.),energy(pts))
            challenger=(int(cnom),int(car>1e-5),max(car,0.),energy(candidate))
            if challenger<current:pts,ar,nominal=candidate,car,cnom
    initialization_seconds=time.perf_counter()-init_started
    best=pts.copy();best_ar=ar;best_nominal=nominal
    def rank(p,r,collision): return (int(collision),int(r>1e-5),max(r,0.),energy(p))
    setup_seconds=time.perf_counter()-started
    trace=[]; solve_seconds=0.;geometry_seconds=0.; radius=trust_radius
    for iteration in range(max_iterations):
        remain=budget_seconds-(time.perf_counter()-started)
        if remain<=.002: break
        x=pts[1:-1].reshape(-1)
        gt=time.perf_counter()
        if len(bb): vals,grads=continuous_losses(pts,centers,half,True)
        else: vals=np.full((samples,1,segs),-1e6);grads=np.zeros(vals.shape+(d,))
        geometry_seconds+=time.perf_counter()-gt
        # Every sample/obstacle/segment is retained in the epigraph linearization,
        # rather than linearizing only the currently worst obstacle.
        flat=grads.reshape(-1,d); c=vals.reshape(-1); rows=len(c)
        owner=np.repeat(np.arange(samples),vals.shape[1]*vals.shape[2])
        riskA=sparse.hstack((sparse.csc_matrix(flat),sparse.csc_matrix((-np.ones(rows),(np.arange(rows),owner)),shape=(rows,samples)), -np.ones((rows,1)),np.zeros((rows,1))),format='csc')
        aggregate=np.zeros((1,n));aggregate[0,yi]=1/(samples*alpha);aggregate[0,ti]=1;aggregate[0,si]=-1
        identity=sparse.eye(n,format='csc')
        lower=np.r_[np.maximum(np.tile(limits[0],free_waypoints),x-radius),np.zeros(samples),-np.inf,0.]
        upper=np.r_[np.minimum(np.tile(limits[1],free_waypoints),x+radius),np.full(samples,np.inf),np.inf,np.inf]
        A=sparse.vstack((riskA,sparse.csc_matrix(aggregate),identity),format='csc')
        lo=np.r_[np.full(rows+1,-np.inf),lower]
        up=np.r_[flat@x-c,0.,upper]
        remain=budget_seconds-(time.perf_counter()-started)
        if remain<=.002:break
        st=time.perf_counter()
        solver=osqp.OSQP()
        solver.setup(P=P,q=q,A=A,l=lo,u=up,verbose=False,eps_abs=1e-5,eps_rel=1e-5,
                     max_iter=10000,polishing=False,time_limit=max(.001,remain),adaptive_rho=True)
        solution=solver.solve(raise_error=False)
        elapsed=time.perf_counter()-st; solve_seconds+=elapsed
        status=solution.info.status
        if solution.x is None or not np.isfinite(solution.x).all():
            trace.append({'iteration':iteration,'status':status,'accepted':False});break
        candidate=np.vstack((start,solution.x[:d].reshape(-1,3),goal))
        candidate[1:-1]=np.clip(candidate[1:-1],limits[0],limits[1])
        car,cnom=evalp(candidate)
        # Merit acceptance/trust-region safeguarding is an explicitly added
        # numerical adaptation for nonsmooth box distance, not upstream code.
        merit_old=energy(pts)+slack_penalty*max(ar,0.)
        merit_new=energy(candidate)+slack_penalty*max(car,0.)
        accepted=merit_new<=merit_old+1e-7
        step=float(np.linalg.norm(candidate-pts))
        if accepted:
            pts=candidate;ar=car;nominal=cnom;radius=min(trust_radius*2,radius*1.25)
        else:radius*=.5
        if rank(candidate,car,cnom)<rank(best,best_ar,best_nominal):
            best=candidate.copy();best_ar=car;best_nominal=cnom
        trace.append({'iteration':iteration,'status':status,'accepted':accepted,'sample_avar':car,
                      'energy':energy(candidate),'qp_slack':float(solution.x[si]),'step':step,'trust_radius':radius,'solve_seconds':elapsed})
        if radius<1e-4 or (accepted and step<1e-5):break
    elapsed=time.perf_counter()-started
    length=float(np.linalg.norm(np.diff(best,axis=0),axis=1).sum())
    return {'method':METHOD,'points':best,'native_representation':'piecewise_linear_first_order_kinematics',
            'controls':np.diff(best,axis=0)*segs,'segment_times':np.linspace(0,1,count),
            'length':length,'control_energy':energy(best),'sample_avar':best_ar,
            'risk_feasible':bool(best_ar<=1e-5),'nominal_collision':bool(best_nominal),
            'status':'feasible' if best_ar<=1e-5 and not best_nominal else 'infeasible_incumbent',
            'seed':int(seed),'sigma':float(sigma),'alpha':float(alpha),'samples':int(samples),
            'budget_seconds':float(budget_seconds),'elapsed_seconds':elapsed,
            'overrun_seconds':max(elapsed-budget_seconds,0.),'setup_seconds':setup_seconds,
            'solve_seconds':solve_seconds,'geometry_seconds':geometry_seconds,'iterations':len(trace),
            'initialization_seconds':initialization_seconds,'initialization_evaluations':initialization_evaluations,
            'trace':trace,'adaptation':'3D static Gaussian boxes; 3 free waypoint kinematic path; energy objective; exact continuous Linf loss; finite-sample AVaR epigraph; SCP/OSQP; added bounded random-arch initializer and trust-region merit safeguard'}
