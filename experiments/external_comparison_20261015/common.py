"""Independent evaluation for continuous polylines and certified cubic chords.

All reported uncertainties translate each entire box once per realization.
This module is deliberately independent of all planner objectives.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'src'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boxes_of(scene):
    return np.asarray([[o['minimum'], o['maximum']] for o in scene['obstacles']], float)


def seed_for(domain, scene_name, root):
    b = hashlib.sha256(f'{domain}|{root}|{scene_name}'.encode()).digest()
    return int.from_bytes(b[:8], 'little')


def collision_free(points, boxes, shifts, margin=0., segment_batch=32):
    """Exact closed-AABB intersection, memory bounded over segments and draws."""
    points, boxes, shifts = map(lambda a: np.asarray(a, float), (points, boxes, shifts))
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('A finite continuous polyline is required')
    result = np.ones(len(shifts), bool)
    if not len(boxes):
        return result
    for begin in range(0, len(points)-1, segment_batch):
        active = np.flatnonzero(result)
        if not len(active):
            break
        p = points[begin:min(len(points),begin+segment_batch+1)]
        start, direction = p[:-1][None,None], np.diff(p,axis=0)[None,None]
        parallel = direction == 0
        den = np.where(parallel, 1., direction)
        # Chunks avoid allocating arrays proportional to the full external bank.
        for offset in range(0, len(active), 256):
            idx = active[offset:offset+256]
            lo = boxes[None,:,None,0,:] + shifts[idx,:,None,:] - margin
            hi = boxes[None,:,None,1,:] + shifts[idx,:,None,:] + margin
            outside = np.any(parallel & ((start < lo) | (start > hi)), axis=-1)
            t1,t2 = (lo-start)/den,(hi-start)/den
            enter = np.maximum(0., np.max(np.where(parallel,-np.inf,np.minimum(t1,t2)),axis=-1))
            leave = np.minimum(1., np.min(np.where(parallel,np.inf,np.maximum(t1,t2)),axis=-1))
            result[idx] = ~np.any((~outside)&(enter<=leave),axis=(1,2))
    return result


@njit(cache=True)
def collision_free_compiled(points, boxes, shifts, margin=0.):
    """Same closed slab predicate, scalar compiled loops and no 4D arrays."""
    free=np.ones(len(shifts),np.bool_)
    for r in range(len(shifts)):
        hit=False
        for j in range(len(boxes)):
            if hit:break
            for i in range(len(points)-1):
                enter=0.;leave=1.;outside=False
                for a in range(3):
                    p=points[i,a];d=points[i+1,a]-p
                    lo=boxes[j,0,a]+shifts[r,j,a]-margin
                    hi=boxes[j,1,a]+shifts[r,j,a]+margin
                    if d==0.:
                        if p<lo or p>hi:outside=True;break
                    else:
                        t0=(lo-p)/d;t1=(hi-p)/d
                        enter=max(enter,min(t0,t1));leave=min(leave,max(t0,t1))
                        if enter>leave:outside=True;break
                if not outside and enter<=leave:
                    hit=True;break
        free[r]=not hit
    return free


def bezier_chords(segments, tolerance=.002, max_depth=24):
    """Subdivide cubic Beziers with a proven uniform L-infinity chord error.

    The chord in cubic Bernstein form has controls [P0, P0+D/3,
    P0+2D/3, P3]. Convexity bounds the curve-minus-chord coordinate
    error by the largest control difference. Expanding every obstacle
    by that scalar error makes collision-free chord checks conservative.
    """
    segments = np.asarray(segments,float)
    if segments.ndim != 3 or segments.shape[1:] != (4,3):
        raise ValueError('Expected (segments,4,3) Bezier controls')
    out, bounds = [segments[0,0]], []
    def visit(b,depth):
        linear = np.linspace(b[0],b[3],4)
        err = float(np.max(np.abs(b-linear)))
        if err<=tolerance:
            out.append(b[-1]);bounds.append(err);return
        if depth>=max_depth:
            raise RuntimeError('Cubic tolerance not reached; do not silently sample')
        a=(b[:-1]+b[1:])/2;c=(a[:-1]+a[1:])/2;d=(c[0]+c[1])/2
        visit(np.array([b[0],a[0],c[0],d]),depth+1)
        visit(np.array([d,c[1],a[2],b[3]]),depth+1)
    for i,b in enumerate(segments):
        if i and not np.allclose(b[0],segments[i-1,-1],rtol=0,atol=1e-8):
            raise ValueError('Discontinuous native curve')
        visit(b,0)
    return np.asarray(out), max(bounds, default=0.)


def ppoly_to_bezier(coefficients, breaks):
    c,t = np.asarray(coefficients,float),np.asarray(breaks,float)
    if c.shape != (4,len(t)-1,3):
        raise ValueError('Cubic PPoly coefficients must have shape (4,nsegments,3)')
    h=np.diff(t)[:,None]
    b0=c[3];b1=b0+c[2]*h/3
    b2=b0+2*c[2]*h/3+c[1]*h*h/3
    b3=b0+c[2]*h+c[1]*h*h+c[0]*h*h*h
    return np.stack([b0,b1,b2,b3],axis=1)


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-radius,mid+radius]


def score(scene,points,*,samples=20000,external_root=2026101503,sigmas=(.6,),curve_error=0.,native_within_bounds=None):
    points=np.asarray(points,float);boxes=boxes_of(scene)
    bound=np.asarray([scene['bounds']['minimum'],scene['bounds']['maximum']],float)
    endpoints=bool(np.allclose(points[0],scene['start'],rtol=0,atol=1e-7) and np.allclose(points[-1],scene['goal'],rtol=0,atol=1e-7))
    # This test is conservative for a certified cubic chord representation.
    in_bounds=bool(np.all(points>=bound[0]+curve_error-1e-9) and np.all(points<=bound[1]-curve_error+1e-9))
    if native_within_bounds is not None:in_bounds=bool(native_within_bounds)
    nominal=endpoints and in_bounds and bool(collision_free(points,boxes,np.zeros((1,len(boxes),3)),margin=curve_error)[0])
    rng=np.random.default_rng(seed_for('external-evaluation-v1',scene['name'],external_root))
    standard=rng.standard_normal((samples,len(boxes),3))
    result={'nominal_success':nominal,'endpoints_correct':endpoints,'within_world_bounds':in_bounds,
            'chord_length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            'curve_chord_error_bound':curve_error,'scored_segments':len(points)-1,
            'external_samples':samples,'external_seed':seed_for('external-evaluation-v1',scene['name'],external_root)}
    bits={}
    for sigma in sigmas:
        free=collision_free_compiled(points,boxes,standard*sigma,curve_error) if endpoints and in_bounds else np.zeros(samples,bool)
        bits[str(sigma)]=np.packbits(free,bitorder='little')
        k=int(free.sum());result[f'cfr_{sigma}']=k/samples
        result[f'cfr_ci_{sigma}']=wilson(k,samples)
        # Nominal success is also required for operational success; retain raw CFR separately.
        result[f'joint_cfr_{sigma}']=k/samples if nominal else 0.
    return result,bits


def generate_bank(output,root_seed,count_per_family,purpose):
    from uav_path_planning.ood_scenarios import make_ood_scenarios
    output=Path(output)
    if (output/'manifest.json').exists():
        return json.loads((output/'manifest.json').read_text(encoding='utf8'))
    scenes=make_ood_scenarios(root_seed,count_per_family)
    items=[]
    for scene in scenes:
        data=scene.to_dict();data['name']=f'{purpose}_{root_seed}_{scene.name}'
        data['metadata']['external_comparison_purpose']=purpose
        p=output/(data['name']+'.json');dump(p,data)
        items.append({'name':data['name'],'family':data['metadata']['family'],'file':p.name,'sha256':sha(p)})
    manifest={'purpose':purpose,'root_seed':root_seed,'maps':items,
              'geometry_only_filter':True,'planner_called_during_generation':False,
              'interpretation':'Fresh instances from existing three-family generator; not a new generator distribution.'}
    dump(output/'manifest.json',manifest);return manifest
