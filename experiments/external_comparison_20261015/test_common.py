import numpy as np
from common import collision_free, collision_free_compiled,bezier_chords,ppoly_to_bezier

def test_closed_segment_box_geometry_and_translation():
    box=np.array([[[0,0,0],[1,1,1]]])
    path=np.array([[-1,.5,.5],[2,.5,.5]])
    shift=np.array([[[0,0,0]],[[0,3,0]]])
    assert collision_free(path,box,shift).tolist()==[False,True]
    assert not collision_free([[-1,1,1],[2,1,1]],box,np.zeros((1,1,3)))[0]
    assert collision_free([[-1,1.01,1],[2,1.01,1]],box,np.zeros((1,1,3)))[0]

def test_vectorized_matches_independent_scalar_box_predicate():
    from uav_path_planning.models import ObstacleBox
    rng=np.random.default_rng(41)
    lo=rng.uniform(-2,2,(4,3));boxes=np.stack([lo,lo+rng.uniform(.1,2,(4,3))],axis=1)
    shifts=rng.normal(size=(80,4,3));points=rng.uniform(-3,3,(6,3))
    expected=[]
    for delta in shifts:
        expected.append(not any(ObstacleBox(b[0]+d,b[1]+d).intersects_segment(a,z)
                               for b,d in zip(boxes,delta) for a,z in zip(points[:-1],points[1:])))
    assert np.array_equal(collision_free(points,boxes,shifts),expected)
    assert np.array_equal(collision_free_compiled(points,boxes,shifts),expected)
    assert np.array_equal(collision_free_compiled(points,boxes,shifts,.025),collision_free(points,boxes,shifts,margin=.025))

def test_cubic_chord_bound_protects_against_hidden_bulge():
    # Endpoints alone miss the cubic arch passing through y=1.5.
    b=np.array([[[0,0,0],[1,2,0],[2,2,0],[3,0,0]]],float)
    points,error=bezier_chords(b,.001)
    box=np.array([[[1.4,1.4,-.1],[1.6,1.6,.1]]])
    assert error<=.001 and len(points)>4
    assert collision_free(b[0,[0,-1]],box,np.zeros((1,1,3)))[0]
    assert not collision_free(points,box,np.zeros((1,1,3)),margin=error)[0]

def test_ppoly_conversion_matches_original_curve():
    from scipy.interpolate import PPoly
    rng=np.random.default_rng(7);c=rng.normal(size=(4,1,3));breaks=np.array([0.,2.])
    b=ppoly_to_bezier(c,breaks)[0];u=np.linspace(0,1,51)[:,None]
    actual=(1-u)**3*b[0]+3*(1-u)**2*u*b[1]+3*(1-u)*u*u*b[2]+u**3*b[3]
    assert np.allclose(actual,PPoly(c,breaks)(2*u[:,0]))
