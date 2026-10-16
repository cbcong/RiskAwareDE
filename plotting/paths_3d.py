"""Approved Figure 2 renderer: four panels, opaque obstacles, dark ground grid.

Orthographic projection of the original cuboids and five-point polylines.
Camera search uses geometric visibility only; no map/seed/path selection.
Visible pieces are solid, pieces behind opaque geometry dashed in the
visibility edition; the strict depth edition omits those hidden pieces.
Colors sampled from user-provided EGO-Planner Figure 8 screenshot.
"""
from pathlib import Path
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/figures'
SOURCE=ROOT/'data/plot_inputs/path_grid_examples.json'
DATA=json.loads(SOURCE.read_text(encoding='utf-8'))
METHODS=[('full','RiskAwareDE','#77FB4C'),('risk_polish_lshade','RT-LS','#EC60F7'),('bare_de','DE','#EA3323')]
FACE_BASE=np.array([57,82,115])/255
FONT='Times New Roman'
plt.rcParams.update({'font.family':FONT,'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.linewidth':.5,'savefig.facecolor':'white','figure.facecolor':'white'})


def camera(elevation,azimuth):
    e,a=np.deg2rad([elevation,azimuth])
    forward=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    right=np.array([-np.sin(a),np.cos(a),0])
    up=np.cross(forward,right)
    return np.stack([right,up,forward],axis=1)


def visible(points,obstacles,cam):
    """Ray to camera intersects no opaque box (whole line segment analytically)."""
    direction=cam[:,2]
    hidden=np.zeros(len(points),bool)
    for o in obstacles:
        lo=np.array(o['minimum']);hi=np.array(o['maximum'])
        safe=np.where(abs(direction)<1e-13,1e-13,direction)
        t0=(lo-points)/safe; t1=(hi-points)/safe
        enter=np.minimum(t0,t1).max(axis=1); leave=np.maximum(t0,t1).min(axis=1)
        hidden |= (leave>=np.maximum(enter,1e-7)) & (leave>1e-7)
    return ~hidden


def samples(path,step=.045):
    # Linear subdivisions are solely rendering/clipping points on existing segments.
    # They introduce no interpolated curvature or new trajectory states.
    points=[];weights=[]
    for a,b in zip(path[:-1],path[1:]):
        n=max(2,int(np.ceil(np.linalg.norm(b-a)/step)))
        t=(np.arange(n)+.5)/n
        points.append(a[None,:]+t[:,None]*(b-a)[None,:]);weights.extend([np.linalg.norm(b-a)/n]*n)
    return np.concatenate(points),np.array(weights)


def faces(o):
    x0,y0,z0=o['minimum'];x1,y1,z1=o['maximum']
    vertices=np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                       [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    definitions=[([0,3,2,1],[0,0,-1]),([4,5,6,7],[0,0,1]),([0,1,5,4],[0,-1,0]),
                 ([1,2,6,5],[1,0,0]),([2,3,7,6],[0,1,0]),([3,0,4,7],[-1,0,0])]
    return [(vertices[ids],np.array(normal)) for ids,normal in definitions]


def panel(ax,e,letter,cam,mode,show_triad=True,ground_grid=False,premium_surface=False,
          grid_color='#D9E2EC',grid_linewidth=.34):
    # The ground plane is an actual z=0 reference plane.  It is only used in
    # the premium horizontal candidate to add depth cues; it is not data and
    # does not hide, alter, or re-route any trajectory.
    if ground_grid:
        floor=np.array([[0,0,0],[20,0,0],[20,20,0],[0,20,0]])@cam
        ax.add_collection(PolyCollection([floor[:,:2]],facecolors=['#F7F9FC'],
                           edgecolors='none',zorder=.2))
        grid=[]
        for value in np.arange(0,20.0001,2.5):
            grid.extend([np.array([[value,0,0],[value,20,0]])@cam,
                         np.array([[0,value,0],[20,value,0]])@cam])
        mesh=LineCollection([line[:,:2] for line in grid],colors=grid_color,
                            linewidths=grid_linewidth,zorder=.4)
        ax.add_collection(mesh)
    objects=[]
    light=np.array([-.3,-.4,1]);light/=np.linalg.norm(light)
    for o in e['map']['obstacles']:
        for verts,normal in faces(o):
            if normal@cam[:,2] <= 0:continue
            pts=verts@cam
            if premium_surface:
                # Separate top and side planes within the same blue family.
                # This encodes face orientation only and leaves the cuboid
                # geometry completely unchanged.
                if normal[2] > .5:
                    color=np.array([86,126,166])/255
                else:
                    intensity=.72+.30*max(0,float(normal@light))
                    color=np.clip(FACE_BASE*intensity,0,1)
            else:
                intensity=.78+.40*max(0,float(normal@light))
                color=np.clip(FACE_BASE*intensity,0,1)
            objects.append((pts[:,2].mean(),pts[:,:2],color))
    objects.sort(key=lambda o:o[0])
    for depth,poly,color in objects:
        ax.add_collection(PolyCollection([poly],facecolors=[color],
                          edgecolors='#DCE8F2' if premium_surface else '#244467',
                          linewidths=.24 if premium_surface else .35,zorder=2))
    visibility={}
    solid_segments=[]
    solid_colors=[]
    solid_depths=[]
    for k,label,color in METHODS[::-1]:
        path=np.array(e['paths'][k]['waypoints'])
        segments=[];hidden=[]
        for a,b in zip(path[:-1],path[1:]):
            n=max(4,int(np.ceil(np.linalg.norm(b-a)/.025)))
            pts=np.linspace(a,b,n+1);xy=pts@cam
            mask=visible((pts[:-1]+pts[1:])/2,e['map']['obstacles'],cam)
            pairs=np.stack([xy[:-1,:2],xy[1:,:2]],axis=1)
            segments.extend(pairs[mask]);hidden.extend(pairs[~mask])
            solid_segments.extend(pairs[mask])
            solid_colors.extend([color]*int(mask.sum()))
            solid_depths.extend(((xy[:-1,2]+xy[1:,2])/2)[mask])
        # Hidden path geometry is not shifted above obstacles. Short dashed
        # pieces identify projection occlusion, as in technical 3D drawings.
        if hidden and mode=='visibility':
            hidden_lines=LineCollection(hidden,colors=color,linewidths=1.2,alpha=.95,zorder=3)
            # Segment-index masking gives physical-space dashes; a dash pattern
            # on separately subdivided segments would accidentally appear solid.
            hidden_lines.set_segments([x for i,x in enumerate(hidden) if (i//10)%2==0])
            hidden_lines.set_path_effects([pe.Stroke(linewidth=2,foreground='white'),pe.Normal()])
            ax.add_collection(hidden_lines)
        pts,weights=samples(path)
        visibility[k]=float(np.average(visible(pts,e['map']['obstacles'],cam),weights=weights))
    # Depth-sort all route pieces jointly so intersections do not systematically
    # privilege one method. Equal-depth coincident portions remain coincident.
    order=np.argsort(solid_depths,kind='stable')
    lines=LineCollection(np.asarray(solid_segments)[order],colors=np.asarray(solid_colors)[order],
                         linewidths=2.0,zorder=4,capstyle='round',joinstyle='round')
    lines.set_path_effects([pe.Stroke(linewidth=2.7,foreground='white',alpha=.80),pe.Normal()])
    ax.add_collection(lines)
    endpoints=np.array([e['map']['start'],e['map']['goal']])@cam
    ax.scatter(*endpoints[0,:2],s=23,c='white',edgecolors='#111111',linewidths=.9,zorder=6)
    ax.scatter(*endpoints[1,:2],s=25,c='#111111',marker='D',edgecolors='white',linewidths=.6,zorder=6)
    corners=np.array([[x,y,z] for x in [0,20] for y in [0,20] for z in [0,10]])@cam
    bounds=np.array([corners[:,:2].min(axis=0),corners[:,:2].max(axis=0)])
    ax.set_xlim(bounds[0,0]-.8,bounds[1,0]+.8);ax.set_ylim(bounds[0,1]-.8,bounds[1,1]+.8)
    ax.set_aspect('equal');ax.axis('off')
    ax.text(.5,-.005,f'({letter})',transform=ax.transAxes,va='top',ha='center',fontsize=12)
    # A single orientation triad is sufficient when the four panels share an
    # identical orthographic camera; duplicating it in every panel wastes
    # column width and makes the horizontal layout visually noisy.
    if show_triad:
        origin=np.array([bounds[0,0]+1.5,bounds[0,1]+2.2]);scale=2
        for vec,label in zip(np.eye(3),['$x$','$y$','$z$']):
            end=origin+(vec@cam)[:2]*scale
            ax.annotate('',end,origin,arrowprops={'arrowstyle':'->','lw':.65,'color':'#555555','shrinkA':0,'shrinkB':0},zorder=8)
            shift=np.array([.12,0]) if label=='$y$' else np.array([-.22,.12]) if label=='$z$' else np.zeros(2)
            ax.text(*(origin+(vec@cam)[:2]*(scale+.65)+shift),label,fontsize=8,ha='center',va='center',color='#333333')
    return visibility


def draw_horizontal(elev=64,az=-45):
    """Render the adopted four-panel layout at its fixed approved camera."""
    cam=camera(elev,az)
    fig,axes=plt.subplots(1,4,figsize=(8.4,2.55))
    fig.subplots_adjust(left=.006,right=.994,top=.992,bottom=.205,wspace=.012)
    metrics=[]
    for i,(ax,e,letter) in enumerate(zip(axes,DATA['examples'],'abcd')):
        metrics.append({'panel':letter,'scene':e['scene'],
                        'visible_fraction':panel(ax,e,letter,cam,'visibility',
                                                  show_triad=False,
                                                  ground_grid=True,
                                                  premium_surface=True,
                                                  grid_color='#B4C4D5',
                                                  grid_linewidth=.46)})
    handles=[Line2D([],[],color=c,lw=3.0,label=n) for k,n,c in METHODS]
    handles.extend([Line2D([],[],marker='o',mfc='white',mec='black',ls='',label='Start',ms=3.8),
                    Line2D([],[],marker='D',color='black',ls='',label='Goal',ms=3.8)])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.012),ncol=5,
               frameon=False,fontsize=10.2,handlelength=1.85,columnspacing=.72,handletextpad=.38)
    base=OUTPUT/'fig06_path_grid'
    base.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(base.with_suffix('.'+ext),dpi=600)
    plt.close(fig)
    return metrics

