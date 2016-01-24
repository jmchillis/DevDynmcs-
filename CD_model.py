#! /usr/bin/env python  # make Python scripts directly executable

# commmented out for now because there is an issue in getting
# import os
# if os.path.isfile(os.environ['PYTHONSTARTUP']):
#     execfile(os.environ['PYTHONSTARTUP'])

#####################
# Key Python modules
#####################
import os, sys
from StringIO import StringIO     # needed for reading text files
from string import *
from datetime import *
# using conventions of scipi commuunity for scipy, numpy and matplotlib
import numpy as np
import numpy.lib.recfunctions          # needed for adding fields to structured arrays
import scipy as sp
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.patches as patches
# mpl.use('macosx',warn=True)          # chose the backend windowing system for ploting
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
from pylab import save
from pythonutils.odict import OrderedDict


####################################################################################################
# Define opportunities to commit CD and the reward sensitivity
####################################################################################################
opprtn = np.linspace(0,1,101)
nRwdPnsh = 12
# parameters of the genotype
rspnPrmExternalizing = np.vstack(([.3,.9,.2],[.7,.3,.4]))  # ,[1,.1,.1]
pExt = [k[0]*sp.stats.norm.cdf(opprtn, k[1], k[2]) for k in rspnPrmExternalizing]
#sensitivity_to_Pns = [sp.stats.uniform.rvs(-.5, 1, size=nRwdPnsh), sp.stats.uniform.rvs(-.5, 1, size=nRwdPnsh)]
sensitivity_to_PnsY = [np.array([1, 1, 1.5, .8, 1, .9, .1, 0, .6, .9, 1.2, 1]),
                      np.aray([.8, .8, .7, 1.4, 1.4 ,.3,.3,.4,.5,.5,.5,.4])
sensitivity_to_PnsX = np.linspace(1,12,12)

#rspnPrmResCntrl = np.vstack(([.3,.9,.2],[.5,.3,.4],[1,.1,.1]))
#sensitivity_to_Rwrd = np.vstack(([.3,.9,.2],[.5,.3,.4],[1,.1,.1]));

# parameters of the social system
# pshmnt_Intensity = np.vstack(([.3,.9,.2],[.5,.3,.4],[1,.1,.1]))
# rwrd_Intensity = np.vstack(([.3,.9,.2],[.5,.3,.4],[1,.1,.1]))

freedom = np.array([.2, .9])
adaptability_pnshX = [np.linspace(1,12,12), np.linspace(1,11,6), np.array([3, 8, 10])
adaptability_pnshY = [np.ones(12)/12, np.array([[.2 .2 .1 .1 .2 .2]), np.array(.8,.1,.1), np.array([3, 8, 10])
resp_coherence = [0, 1]
deg_monitor = [.2, .8]

# adaptability_freedom = [0, .2, 1]

## pshmnt_Likelihood = 1-1/np.exp(
## rwrd_ControlStrength =
## rwrd_Likelihood = 1/pshmnt_Likelihood

#popSampS = sp.stats.uniform.rvs(0, 1, size=100)
#popSampMU = sp.stats.norm.rvs(.8, 2, size=100)
#popSampVAR = sp.stats.norm.rvs(.8, 2, size=100)
#pRst = [k[0]*sp.stats.norm.cdf(opprtn, k[1], k[2]) for k in rspnPrmResCntrl]


clrs = [(0.0, 0.0, 0.0),(0.3, 0.3, 0.3),(0.0, 0.0, 0.0),(0.7, 0.7, 0.7),(1.0, 0.2, 0.2),(0.4, 0.4, 0.4)]
fig = plt.figure(num=1, figsize=(10,10), facecolor='w', edgecolor='k')
ax = fig.add_subplot(2, 2, 1)
plt.axis([0, 1, -.1, 1.1])
# plot freedom
verts = [(0,0), (0,1), (.2, 1), (.2, 0), (0,0)]
codes = [Path.MOVETO,Path.LINETO,Path.LINETO, Path.LINETO,Path.CLOSEPOLY,]
path = Path(verts, codes)
patch = patches.PathPatch(path, facecolor='gray', lw=0)
ax.add_patch(patch)

pG1ext = ax.plot(opprtn,pExt[0], lw=2, c=clrs[0], marker = 'None',  label="A")
pG2ext = ax.plot(opprtn,pExt[1], lw=2, c=clrs[1], marker = 'None',  label="C")
pG3ext = ax.plot(opprtn,pExt[2], ls = '--', lw=2, c=clrs[2], marker = 'None',  label="E")
ax2 = fig.add_subplot(2, 2, 2)
plt.axis([0, 3, -.1, 1.5])
pG1ext = ax.plot(opprtn,pExt[0], lw=2, c=clrs[0], marker = 'None',  label="A")
pG2ext = ax.plot(opprtn,pExt[1], lw=2, c=clrs[1], marker = 'None',  label="C")
pG3ext = ax.plot(opprtn,pExt[2], ls = '--', lw=2, c=clrs[2], marker = 'None',  label="E")


pAl = ax2.plot(vals2,(coefAl[0] + coefAl[1]*vals2)**2, lw=2, c=clrs[0], marker = 'None',  label="A")
pAu = ax2.plot(vals2,(coefAu[0] + coefAu[1]*vals2)**2, lw=2, c=clrs[0], marker = 'None',  label="A")
ax3 = fig.add_subplot(2, 2, 3)
plt.axis([0, 3, -.1, 1.5])
pCl = ax3.plot(vals2,(coefCl[0] + coefCl[1]*vals2)**2, lw=2, c=clrs[1], marker = 'None',  label="C")
pCu = ax3.plot(vals2,(coefCu[0] + coefCu[1]*vals2)**2, lw=2, c=clrs[1], marker = 'None',  label="C")
ax4 = fig.add_subplot(2, 2,4)
plt.axis([0, 3, -.1, 1.5])
pEl = ax4.plot(vals2,(coefEl[0] + coefEl[1]*vals2)**2, ls = '--', lw=2, c=clrs[2], marker = 'None',  label="E")
pEu = ax4.plot(vals2,(coefEu[0] + coefEu[1]*vals2)**2, ls = '--', lw=2, c=clrs[2], marker = 'None',  label="E")
ax3.set_xlabel('Parent-Child Conflict',{'fontsize':16})
ax4.set_xlabel('Parent-Child Conflict',{'fontsize':16})
ax.set_ylabel('Unstandardized variance ',{'fontsize':16})
ax3.set_ylabel('Unstandardized variance ',{'fontsize':16})
ax.set_title('Fit of AE moderation model',{'fontsize':16})
ax2.set_title('95% confidence intervals for A',{'fontsize':16})
ax3.set_title('95% confidence intervals for C',{'fontsize':16})
ax4.set_title('95% confidence intervals for E',{'fontsize':16})
ax.legend(("A","C","E"),'upper left')
plt.savefig('GxEhmwkFig.png',format='png')

h = plt.plot(x, rv.pdf(x))
