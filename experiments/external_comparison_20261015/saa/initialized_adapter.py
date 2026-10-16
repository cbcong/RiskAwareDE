"""Declared A*-initialized geometric SAA/AVaR adaptation.

Initialization is charged to the same wall-time allowance. It uses nominal
geometry only and preserves bends required for a collision-free initial path.
This is not the original six-state drone experiment or an unmodified solver.
"""
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parents[2]/'src'))
from saa.adapter import plan as optimize
from common import collision_free


def plan(start,goal,boxes,bounds,seed=0,budget_seconds=6.,sigma=.6,
         alpha=.1,samples=32,min_free_waypoints=5,**options):
    from uav_path_planning.scenarios import Scenario3D
    from uav_path_planning.benchmarks import BenchmarkConfig,_result_for_astar
    started=time.perf_counter()
    boxes=np.asarray(boxes,float);bounds=np.asarray(bounds,float)
    scene=Scenario3D.from_dict({'name':'saa_initialization','start':list(start),'goal':list(goal),
        'bounds':{'minimum':bounds[0].tolist(),'maximum':bounds[1].tolist()},
        'obstacles':[{'minimum':b[0].tolist(),'maximum':b[1].tolist()} for b in boxes]})
    base=_result_for_astar(scene,BenchmarkConfig(grid_resolution=1.,safety_margin=0.))
    if base.path is None:
        return {'method':'SAA-AVaR-3D (A*-initialized geometric adaptation)','points':None,
                'status':'initialization_failed','risk_feasible':False,'nominal_collision':True,
                'elapsed_seconds':time.perf_counter()-started}
    raw=np.asarray(base.path.waypoints,float)
    chosen=[raw[0]];i=0
    while i<len(raw)-1:
        for j in range(len(raw)-1,i,-1):
            if collision_free([raw[i],raw[j]],boxes,np.zeros((1,len(boxes),3)))[0]:break
        chosen.append(raw[j]);i=j
    points=np.asarray(chosen)
    while len(points)<min_free_waypoints+2:
        k=int(np.argmax(np.linalg.norm(np.diff(points,axis=0),axis=1)))
        points=np.insert(points,k+1,(points[k]+points[k+1])/2,axis=0)
    initialization_seconds=time.perf_counter()-started
    remaining=budget_seconds-initialization_seconds
    if remaining<=.01:
        return {'method':'SAA-AVaR-3D (A*-initialized geometric adaptation)','points':points,
                'status':'initialization_budget_exhausted','risk_feasible':False,'nominal_collision':False,
                'elapsed_seconds':time.perf_counter()-started,'initialization_seconds':initialization_seconds}
    result=optimize(start,goal,boxes,bounds,seed=seed,budget_seconds=remaining,sigma=sigma,
                    alpha=alpha,samples=samples,free_waypoints=len(points)-2,initial_points=points,**options)
    result['solver_elapsed_seconds']=result['elapsed_seconds']
    result['elapsed_seconds']=time.perf_counter()-started
    result['budget_seconds']=budget_seconds
    result['overrun_seconds']=max(0.,result['elapsed_seconds']-budget_seconds)
    result['method']='SAA-AVaR-3D (A*-initialized geometric adaptation)'
    result['grid_initialization']={'seconds':initialization_seconds,'grid_resolution':1.,'padding':0.,
        'raw_waypoints':len(raw),'free_waypoints':len(points)-2,'expanded_nodes':base.evaluations,
        'initial_points':points.tolist(),'selection_basis':'nominal geometry only, no external samples'}
    return result
