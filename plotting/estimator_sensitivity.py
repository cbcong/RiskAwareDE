"""Current estimator and sensitivity plots; frozen data, unchanged statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, NullFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/plot_inputs'; S=D
OUTPUT=ROOT/'outputs/figures'
COL={'full':'#27408B','bare_de':'#777777','bare_lshade':'#202020',
 'risk_polish_de':'#704849','risk_polish_jade':'#6B617A',
 'risk_polish_shade':'#547080','risk_polish_lshade':'#404040',
 'risk_polish_pso':'#766653'}
KEYS=list(COL)
MARK={'full':'o','bare_de':'s','bare_lshade':'^','risk_polish_de':'s',
 'risk_polish_jade':'D','risk_polish_shade':'^','risk_polish_lshade':'v','risk_polish_pso':'P'}
NAME={'full':'RiskAwareDE','bare_de':'DE','bare_lshade':'L-SHADE',
 'risk_polish_de':'RT-DE','risk_polish_jade':'RT-JADE','risk_polish_shade':'RT-SHADE',
 'risk_polish_lshade':'RT-LS','risk_polish_pso':'RT-PSO'}
INK='#171717';GREY='#777777';LIGHT='#E5E5E5';ORANGE='#704849'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral'],
 'mathtext.fontset':'stix','font.size':8.5,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'axes.edgecolor':INK,'axes.linewidth':.6,'lines.linewidth':1,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':330,
 'axes.spines.top':True,'axes.spines.right':True,'xtick.major.width':.6,'ytick.major.width':.6,
 'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3})

def read(name,sd=False):return pd.read_csv((S if sd else D)/name)


def save(fig,name,sup=False):
 name={'figS02_sensitivity':'fig09_sensitivity','figS04_estimator':'fig08_estimator'}.get(name,name)
 p=OUTPUT/name
 p.parent.mkdir(parents=True,exist_ok=True)
 for ext in ['pdf','png','svg']:fig.savefig(p.with_suffix('.'+ext),bbox_inches='tight',pad_inches=.08,facecolor='white')
 plt.close(fig)


def panel(ax,letter,title):
 # Combine the panel identifier with the existing x-axis label rather than
 # adding a second descriptive row or a heading above the data.
 ax.set_xlabel(f'({letter}) {ax.get_xlabel()}')


def style(ax):ax.grid(False);ax.set_axisbelow(True)


def sensitivity():
 r=read('P3_local_sensitivity_summary.csv');fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
 for ax,sub,x,label,letter in [(axs[0],r[r.nfe==820].sort_values('beta'),'beta',r'Soft-risk weight, $\beta$','a'),(axs[1],r[r.beta==.05].sort_values('nfe'),'nfe','Total evaluations','b')]:
  twin=ax.twinx();ax.plot(sub[x],sub.length_feasible,'o-',c=COL['full'],ms=4);twin.plot(sub[x],sub.cfr06*100,'s--',c=ORANGE,ms=4)
  ax.set_xlabel(label);ax.set_ylabel('Mean path length',color=COL['full']);twin.set_ylabel(r'CFR at $\sigma=0.6$ (%)',color=ORANGE);twin.spines['right'].set_visible(True);twin.spines['right'].set_color(ORANGE);twin.tick_params(axis='y',labelcolor=ORANGE);ax.tick_params(axis='y',labelcolor=COL['full']);style(ax);panel(ax,letter,'Weight sensitivity' if x=='beta' else 'Budget sensitivity')
  ax.set_xticks(sub[x]);ax.set_ylim(26.92,27.15);twin.set_ylim(87,91.5)
  # Both panels use the same two y ranges. Show each scale once on the
  # outside, leaving the middle clear without changing any normalization.
  if letter=='a':
   twin.set_ylabel('');twin.tick_params(axis='y',right=False,labelright=False)
  else:
   ax.set_ylabel('');ax.tick_params(axis='y',left=False,labelleft=False)
 fig.subplots_adjust(left=.10,right=.91,bottom=.23,top=.83,wspace=.14);save(fig,'figS02_sensitivity',True)


def estimator():
 r=read('P0_metrics.csv');r=r[r.K==16];p=r.pivot(index=['map_root','scene','method'],columns='estimator',values='empirical_variance')
 fig,axs=plt.subplots(1,2,figsize=(7.2,2.7));ax=axs[0]
 for method,c,label in [('full',COL['full'],'Main-method paths'),('bare_lshade',COL['risk_polish_lshade'],'L-SHADE paths')]:
  if method not in p.index.get_level_values('method'):continue
  a=p.xs(method,level='method');ax.scatter(a.binary*1000,a.conditional*1000,c=c,s=22,label=label,alpha=.8,edgecolors='white',lw=.4)
 ax.plot([0,25],[0,25],ls='--',lw=.7,c=GREY);ax.set(xlim=(0,25),ylim=(0,25),xlabel=r'Binary variance ($\times 10^{-3}$)',ylabel=r'Conditional variance ($\times 10^{-3}$)');panel(ax,'a','Paired fixed-path variance');ax.legend(frameon=False,fontsize=7)
 ax=axs[1];r=read('P0_estimator_summary.csv');r=r[r.path_origin=='full']
 for est,c,label in [('binary',ORANGE,'IID binary'),('conditional',COL['full'],'Conditional')]:
  a=r[r.estimator==est].sort_values('K');ax.plot(a.K,a.mean_gate_disagreement_vs_reference*100,'o-',c=c,ms=4,label=label)
 ax.set(xlabel='Internal samples, K',ylabel='Threshold disagreement (%)',xscale='log',ylim=(10,50));ax.set_xticks([8,16,32,64],['8','16','32','64']);ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter());panel(ax,'b','Decision stability');style(ax);ax.legend(frameon=False,fontsize=7)
 # Distinct metrics retain distinct scales, positioned on the two outside
 # edges so the central gutter does not need to hold a vertical axis title.
 ax.yaxis.tick_right();ax.yaxis.set_label_position('right')
 ax.tick_params(axis='y',left=False,right=True)
 fig.subplots_adjust(left=.095,right=.905,bottom=.21,top=.85,wspace=.14);save(fig,'figS04_estimator',True)

