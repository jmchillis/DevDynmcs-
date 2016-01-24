import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats

def PlotModelDemo(opportintyFcn,ResistnceToControl=[],sens_To_Reward=[],sens_To_Punish=[],immtatePredisp=[]) :

    ####################################################################################################
    ## PLOT CP mean counts as function of age
    ####################################################################################################
    clrs = [(1.0, 0.0, 0.0),(0.0, 0.0, 0.0),(1.0, 0.7, 0.7),(0.7, 0.7, 0.7),(1.0, 0.2, 0.2),(0.4, 0.4, 0.4)]
    fig = plt.figure(num=1, figsize=(12,12), facecolor='w', edgecolor='k')
    ax = fig.add_subplot(2, 2, 1)
    plt.axis([-.5, 20, .3,1])
    ax.plot(unqAges,coefM[0][0]+coefM[0][1]*unqAges,
            ms=0, c=clrs[5], marker='none',lw=2,ls='-')
    ax.plot(unqAges,coefF[0][0]+coefF[0][1]*unqAges,
            ms=0, c=clrs[4], marker='none',lw=2,ls='-')
    #errM = ax.plot(agesM,muCPxAgeM,lw=0, c=clrs[1], marker = 'o', mfc = clrs[1], mec = clrs[1],  label="Males")
    #errF = ax.plot(agesF,muCPxAgeF,lw=0, c=clrs[0], marker = 'o', mfc = clrs[0], mec = clrs[0], label="Females")
    errM = ax.errorbar(agesM,muCPxAgeM,yerr=stdeCPxAgeM,
             lw=0, c=clrs[1], marker = 'o', mfc = clrs[1], mec = clrs[1], ecolor='k', elinewidth=2, capsize=4, label="Males")
    errF = ax.errorbar(agesF,muCPxAgeF,yerr=stdeCPxAgeF,
             lw=0, c=clrs[0], marker = 'o', mfc = clrs[0], mec = clrs[0], elinewidth=2, capsize=4,label="Females")
    ax.set_ylabel('mean ln(CP) ',{'fontsize':16})
    ax.set_title('Linear Fit',{'fontsize':16})

    ax.legend() # ('males', 'females') )

    ax4 = fig.add_subplot(2, 2, 2)
    plt.axis([-.5, 20, .3,1])
    if len(coefM_nlm) != 0 :
        if len(coefM_nlm) >= 7  :    # in case where factor for number of siblings is used
            for k in [3,4,5,6] :
                ax4.plot(unqAges,coefM_nlm[0]+coefM_nlm[k]+coefM_nlm[7]+coefM_nlm[1]*unqAges+coefM_nlm[2]*unqAges**2,
                         ms=0, c=clrs[3], marker='none',lw=2,ls='-')
        elif len(coefM_nlm) == 3 :
            ax4.plot(unqAges,coefM_nlm[0]+coefM_nlm[1]*unqAges+coefM_nlm[2]*unqAges**2,
                     ms=0, c=clrs[3], marker='none',lw=2,ls='-')

    if len(coefF_nlm) != 0 :
        if len(coefM_nlm) >= 7 :    # in case where factor for number of siblings is used
            for k in [3,4,5,6] :
                ax4.plot(unqAges,coefF_nlm[0]+coefF_nlm[k]+coefF_nlm[7]+coefF_nlm[1]*unqAges+coefF_nlm[2]*unqAges**2,
                         ms=0, c=clrs[2], marker='none',lw=2,ls='-')
        elif len(coefM_nlm) == 3 :
            ax4.plot(unqAges,coefF_nlm[0]+coefF_nlm[1]*unqAges+coefF_nlm[2]*(unqAges**2),
                     ms=0, c=clrs[2], marker='none',lw=2,ls='-')

    #errM = ax.plot(agesM,muCPxAgeM,lw=0, c=clrs[1], marker = 'o', mfc = clrs[1], mec = clrs[1],  label="Males")
    #errF = ax.plot(agesF,muCPxAgeF,lw=0, c=clrs[0], marker = 'o', mfc = clrs[0], mec = clrs[0], label="Females")
    errM = ax4.errorbar(agesM,muCPxAgeM,yerr=stdeCPxAgeM,
             lw=0, c=clrs[1], marker = 'o', mfc = clrs[1], mec = clrs[1], ecolor='k', elinewidth=2, capsize=4, label="Males")
    errF = ax4.errorbar(agesF,muCPxAgeF,yerr=stdeCPxAgeF,
             lw=0, c=clrs[0], marker = 'o', mfc = clrs[0], mec = clrs[0], elinewidth=2, capsize=4,label="Females")
    ax4.set_title('Polynomial 2 Fit',{'fontsize':16})

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.axis([-.5, 20, -.5, .5])
    ax2.plot(np.array([0,20]),np.array([0,0]),marker='none',lw=2,c=clrs[3])
    ax2.errorbar(agesM,muCPxAgeMr,yerr=stdeCPxAgeMr,xerr=None,
             lw=0, c=clrs[1], marker = 'o', mfc = clrs[1], mec = clrs[1], fmt='-', ecolor='k', elinewidth=2, capsize=4)
    ax2.errorbar(agesF,muCPxAgeFr,yerr=stdeCPxAgeFr,xerr=None,
             lw=0, c=clrs[0], marker = 'o', mfc = clrs[0], mec = clrs[0], fmt='-', ecolor=clrs[0], elinewidth=2, capsize=4)
    ax2.set_ylabel('Mean Residuals lin. fit',{'fontsize':16})
    ax2.set_xlabel('Twin Age',{'fontsize':16})

    if len(coefM_nlm) != 0 :
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.axis([-.5, 20, -.5, .5])
        ax3.plot(np.array([0,20]),np.array([0,0]),marker='none',lw=2,c=clrs[3])
        ax3.errorbar(agesM,muCPxAgeMr_nl,yerr=stdeCPxAgeMr_nl,xerr=None,
                 lw=0, c=clrs[1], marker = 'o', mfc = clrs[1], mec = clrs[1], fmt='-', ecolor='k', elinewidth=2, capsize=4)
        ax3.errorbar(agesF,muCPxAgeFr_nl,yerr=stdeCPxAgeFr_nl,xerr=None,
                 lw=0, c=clrs[0], marker = 'o', mfc = clrs[0], mec = clrs[0], fmt='-', ecolor=clrs[0], elinewidth=2, capsize=4)
        ax3.set_xlabel('Twin Age',{'fontsize':16})

    plt.savefig('figures/CPfcnAGEandSex.pdf',format='pdf')
