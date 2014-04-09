# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:56:21 2014

@author: falvarez
"""

#!/usr/bin/env python

import numpy as np
import multiprocessing as mp
import os

try:
    import prettyplotlib as ppl
except:
    ppl_flag = False
else:
    ppl_flag = True

try:
    from sklearn.metrics import roc_curve,auc
except:
    scikit_flag = False
else:
    scikit_flag = True

def _block_bootstrap(fcst,verif,levs,bs_start,bs_end,n_blocks,block_length,fcst_num,lock,reliability):
    dummy_reliability = np.zeros((int(bs_end-bs_start),len(levs)))        
    for btstrp in xrange(0,int(bs_end-bs_start)):
        dummy_fcst = np.zeros((n_blocks,block_length,))
        dummy_verif = np.zeros((n_blocks,block_length,))
        for blk in xrange(0,n_blocks):
            # --- get random samples
            rand_int = np.random.randint(0,fcst_num-block_length)
            # --- extract block of fcst and verif
            dummy_fcst[blk,:] = fcst[rand_int:(rand_int+block_length)]
            dummy_verif[blk,:] = verif[rand_int:(rand_int+block_length)]
        dummy_fcst = dummy_fcst.reshape(-1)
        dummy_verif = dummy_verif.reshape(-1)
        for i in xrange(len(levs)):
            prob1 = levs[i]+2.5
            prob2 = levs[i]-2.5
            test1 = (dummy_fcst < prob1)
            test2 = (dummy_fcst >= prob2)
            testfreq = 1.0*test1*test2
            testverif = (dummy_verif*testfreq)
            totfreq = np.sum(testfreq)
            obfreq = np.sum(testverif)
            # --- Find reliability from the samples
            dummy_reliability[btstrp,i] = 100.*(obfreq/totfreq)
    with lock:
        print "writing process {}".format(os.getpid())
        reliability[int(bs_start*len(levs)):int(bs_end*len(levs))] = dummy_reliability.reshape(-1)

def _reliability_block_bootstrap(fcst,verif,levs,n_bootstraps,n_blocks,block_length,alpha):
    """
    Function to find bootstrap confidence intervals for reliability
    """

    high = np.zeros(len(levs)) # --- High end of confidence interval
    low = np.zeros(len(levs)) # --- Low end of confidence interval
    nproc = mp.cpu_count() # --- Need to find out how many cores we're working with
    t_nums = np.floor(np.linspace(0,n_bootstraps,nproc+1)) # --- Basically, the starting/ending index for each job
    arr = mp.Array('d',n_bootstraps*len(levs),lock=True) # --- The eventual array we will be sending into the netCDF4 file
    mp_fcst = mp.Array('d',fcst.shape[0],lock=False)
    mp_verif = mp.Array('d',fcst.shape[0],lock=False)
    mp_fcst[:] = fcst
    mp_verif[:] = verif
    fcst_num = fcst.shape[0]
    processes = [] # --- List of jobs     
    print "Setting up and running multiprocessing block bootstrap function..."
    for i in range(nproc):
         bs_start = t_nums[i]
         bs_end = t_nums[i+1]
         lock = mp.Lock()
         p = mp.Process(target=_block_bootstrap,args=(mp_fcst,mp_verif,levs,bs_start,bs_end,n_blocks,block_length,fcst_num,lock,arr))
         p.start()
         processes.append(p)   
    for i in processes:
         i.join()
         
    reliability = np.frombuffer(arr.get_obj())
    reliability = reliability.reshape(n_bootstraps,len(levs))
    
    # --- Sort and find CIs
    stat = np.sort(reliability,axis=0)
    print "Calculating confidence intervals..."
    for i in xrange(len(levs)):
        temp_stats = np.ma.masked_invalid(stat[:,i]).compressed()
        try:
            low[i] = temp_stats[int((alpha/2.0)*temp_stats.shape[0])]
        except IndexError:
            continue
        else:
            low[i] = temp_stats[int((alpha/2.0)*temp_stats.shape[0])]
            high[i] = temp_stats[int((1-alpha/2.0)*temp_stats.shape[0])]
    return low,high
    
def reliability_diagram(fig,fcst,verif,**kwargs):
    """
        A function that generates reliability diagrams.
        
        Input:
        
        fig - Some matplotlib figure object
        fcst - a 1-D NumPy array of forecast probabilities
        verif - a 1-D NumPy array of binary verification values
        
        Optional Arguments:
        
        levs - 1-D NumPy array of probability values to measure reliability (default: np.arange(0,101,5))
        lev_int - Value to act as window to reduce sampling error within reliability (default: .5*(levs[1]-levs[0]))
        bootstrap - Should we find confidence intervals via block bootstrap? (default: False)
        n_bootstraps - Number of times to sample with replacement (default: 1000)
        n_blocks - Number of blocks to sample (default: 1000)
        block_length - Size of block to sample (default: 100)
        ci_alpha - Confidence Interval (default: .05, allows for 5% and 95% CIs)
        
    """    
    levs = kwargs.get('levs',np.arange(0,101,5))
    lev_int = kwargs.get('lev_int',.5*(levs[1]-levs[0]))
    bootstrap = kwargs.get('bootstrap',False)
    n_bootstraps = kwargs.get('n_bootstraps',1000)
    n_blocks = kwargs.get('n_blocks',1000)
    block_length = kwargs.get('block_length',100)
    ci_alpha = kwargs.get('ci_alpha',.05)

    if fcst.shape[0] != verif.shape[0]:
        raise Exception('Forecast and verification arrays need to be the same shape!\nForecast array shape: {}\nVerification array shape: {}'.format(fcst.shape[0],verif.shape[0]))

    # --- Reliability
    total_frequency = np.zeros(len(levs))
    observed_frequency = np.zeros(len(levs))
    reliability = np.zeros(len(levs),'f')
    for i in xrange(len(levs)):
        prob1 = levs[i]+lev_int
        prob2 = levs[i]-lev_int
        test1 = (fcst < prob1)
        test2 = (fcst >= prob2)
        #print test1,test2
        testfreq = 1.0*test1*test2
        testverif = (verif*testfreq)
        total_frequency[i] = np.sum(testfreq)
        observed_frequency[i] = np.sum(testverif)
        reliability[i] = 100.*observed_frequency[i]/total_frequency[i]
    if bootstrap:
        low,high = _reliability_block_bootstrap(fcst,verif,levs,n_bootstraps,n_blocks,block_length,ci_alpha)
    # --- BSS

    # --- freq chart
    total_sum=fcst.shape[0]
    frequency = np.zeros(len(levs),'f')
    for i in xrange(len(levs)):
        frequency[i] = total_frequency[i]/total_sum
    
    if ppl:
        a2 = fig.add_axes([.12,.12,.83,.8])
        a3 = fig.add_axes([.25,.65,.34,.21])
        ppl.bar(a3,levs,frequency,width=5,linewidth=2,bottom=0.00001,log=True,grid='y',color='red',edgecolor='black',align='center')
        a3.set_title('Frequency of usage',fontsize=9.5)
        a3.set_xlabel('Probability',fontsize=9)
        a3.set_ylabel('Frequency',fontsize=9)
        a3.set_xlim(-3,103)
        a3.set_ylim(0.0,1.)
        if bootstrap:
            a2.errorbar(levs,reliability,yerr=[reliability-low,high-reliability],ecolor='k',fmt='ro')
        ppl.plot(a2,levs,levs,linestyle='--',color='k')
        ppl.plot(a2,levs,reliability,linestyle='-',marker='o',color='red')
        a2.xaxis.set_ticks(levs)
        a2.yaxis.set_ticks(levs)
        a2.minorticks_on()
        a2.set_ylabel('Observed Relative Frequency')
        a2.set_xlabel('Forecast Probability')
        a2.set_title('Reliability',fontsize=12)
        a2.set_ylim(0,levs[-1])
        a2.set_xlim(0,levs[-1])

    else:
        a2 = fig.add_axes([.12,.12,.83,.8])
        a3 = fig.add_axes([.25,.65,.34,.21])
        a3.bar(levs,frequency,width=5,linewidth=2,bottom=0.00001,log=True,color='red',edgecolor='black',align='center')
        a3.set_title('Frequency of usage',fontsize=9.5)
        a3.set_xlabel('Probability',fontsize=9)
        a3.set_ylabel('Frequency',fontsize=9)
        a3.set_xlim(-3,103)
        a3.set_ylim(0.0,1.)
        if bootstrap:
            a2.errorbar(levs,reliability,yerr=[reliability-low,high-reliability],ecolor='k',fmt='ro')
        a2.plot(levs,levs,linestyle='--',color='k')
        a2.plot(levs,reliability,linestyle='-',marker='o',color='r')
        a2.xaxis.set_ticks(levs)
        a2.yaxis.set_ticks(levs)
        a2.minorticks_on()
        a2.set_ylabel('Observed Relative Frequency')
        a2.set_xlabel('Forecast Probability')
        a2.set_title('Reliability',fontsize=12)
        a2.set_ylim(0,levs[-1])
        a2.set_xlim(0,levs[-1])

    return fig
    
def relia_roc(fig,fcst,verif,**kwargs):
    """
        A function that generates a reliability diagram and ROC curve. Since ROC is conditioned on the observations and
        a reliability diagram is conditioned on the forecasts, they are good companion diagrams.
        
        Input:
        
        fig - Some matplotlib figure object
        fcst - a 1-D NumPy array of forecast probabilities
        verif - a 1-D NumPy array of binary verification values
        
        Optional Arguments:
        
        levs - 1-D NumPy array of probability values to measure reliability (default: np.arange(0,101,5))
        lev_int - Value to act as window to reduce sampling error within reliability (default: .5*(levs[1]-levs[0]))
        bootstrap - Should we find confidence intervals via block bootstrap? (default: False)
        n_bootstraps - Number of times to sample with replacement (default: 1000)
        n_blocks - Number of blocks to sample (default: 1000)
        block_length - Size of block to sample (default: 100)
        ci_alpha - Confidence Interval (default: .05, allows for 5% and 95% CIs)
        
    """    
    levs = kwargs.get('levs',np.arange(0,101,5))
    lev_int = kwargs.get('lev_int',2.5)
    bootstrap = kwargs.get('bootstrap',False)
    n_bootstraps = kwargs.get('n_bootstraps',1000)
    n_blocks = kwargs.get('n_blocks',1000)
    block_length = kwargs.get('block_length',100)
    ci_alpha = kwargs.get('ci_alpha',.05)

    if fcst.shape[0] != verif.shape[0]:
        raise Exception('Forecast and verification arrays need to be the same shape!\nForecast array shape: {}\nVerification array shape: {}'.format(fcst.shape[0],verif.shape[0]))

    if not scikit_flag:
        raise Exception('We need the scikit-learn module istalled in order to run this test!')
        
    # --- Reliability
    total_frequency = np.zeros(len(levs))
    observed_frequency = np.zeros(len(levs))
    reliability = np.zeros(len(levs),'f')
    for i in xrange(len(levs)):
        prob1 = levs[i]+lev_int
        prob2 = levs[i]-lev_int
        test1 = (fcst < prob1)
        test2 = (fcst >= prob2)
        #print test1,test2
        testfreq = 1.0*test1*test2
        testverif = (verif*testfreq)
        total_frequency[i] = np.sum(testfreq)
        observed_frequency[i] = np.sum(testverif)
        reliability[i] = 100.*observed_frequency[i]/total_frequency[i]
    if bootstrap:
        low,high = _reliability_block_bootstrap(fcst,verif,levs,n_bootstraps,n_blocks,block_length,ci_alpha)
    # --- BSS

    # --- freq chart
    total_sum=fcst.shape[0]
    frequency = np.zeros(len(levs),'f')
    for i in xrange(len(levs)):
        frequency[i] = total_frequency[i]/total_sum
    
    
    fcst = np.rint(fcst)
    # --- ROC curve
    fpr,tpr,thresholds = roc_curve(verif,fcst)
    roc_auc = auc(fpr,tpr)
    
    if ppl:

        a1 = fig.add_axes([.06,.08,.40,.8])
        ppl.plot(a1,fpr,tpr,color='b',label='ROC curve (area = %0.2f)' % roc_auc)
        ppl.plot(a1,[0,1],[0,1],linestyle='--',color='k')
        a1.xaxis.set_ticks([0,.2,.4,.6,.8,1.])
        a1.yaxis.set_ticks([0,.2,.4,.6,.8,1.])
        a1.minorticks_on()
        a1.set_ylabel('Hit Rate')
        a1.set_xlabel('False Alarm Rate')
        a1.set_title('ROC Curve',fontsize=12)
        a1.set_ylim([0.0,1.0])
        a1.set_xlim([0.0,1.0])
        ppl.legend(loc="lower right")
        
        a2 = fig.add_axes([.55,.08,.40,.8])
        a3 = fig.add_axes([.625,.625,.17,.21])
        ppl.bar(a3,levs,frequency,width=5,linewidth=2,bottom=0.00001,log=True,grid='y',color='red',edgecolor='black',align='center')
        a3.set_title('Frequency of usage',fontsize=9.5)
        a3.set_xlabel('Probability',fontsize=9)
        a3.set_ylabel('Frequency',fontsize=9)
        a3.set_xlim(-3,103)
        a3.set_ylim(0.0,1.)
        #ppl.plot(a2,levs,reliability,linestyle='-',marker='o',color='r')
        if bootstrap:
            a2.errorbar(levs,reliability,yerr=[reliability-low,high-reliability],ecolor='k',fmt='ro')
        ppl.plot(a2,levs,levs,linestyle='--',color='k')
        ppl.plot(a2,levs,reliability,linestyle='-',marker='o',color='r')
        a2.xaxis.set_ticks(levs)
        a2.yaxis.set_ticks(levs)
        a2.minorticks_on()
        a2.set_ylabel('Observed Relative Frequency')
        a2.set_xlabel('Forecast Probability')
        a2.set_title('Reliability',fontsize=12)
        a2.set_ylim(0,levs[-1])
        a2.set_xlim(0,levs[-1])

    else:
        
        a1 = fig.add_axes([.06,.08,.40,.8])
        a1.plot(fpr,tpr,color='b',label='ROC curve (area = %0.2f)' % roc_auc)
        a1.plot([0,1],[0,1],linestyle='--',color='k')
        a1.xaxis.set_ticks([0,.2,.4,.6,.8,1.])
        a1.yaxis.set_ticks([0,.2,.4,.6,.8,1.])
        a1.minorticks_on()
        a1.set_ylabel('Hit Rate')
        a1.set_xlabel('False Alarm Rate')
        a1.set_title('ROC Curve',fontsize=12)
        a1.set_ylim([0.0,1.0])
        a1.set_xlim([0.0,1.0])
        a1.legend(loc="lower right")
        
        a2 = fig.add_axes([.55,.08,.40,.8])
        a3 = fig.add_axes([.625,.625,.17,.21])
        a3.bar(levs,frequency,width=5,linewidth=2,bottom=0.00001,log=True,color='red',edgecolor='black',align='center')
        a3.set_title('Frequency of usage',fontsize=9.5)
        a3.set_xlabel('Probability',fontsize=9)
        a3.set_ylabel('Frequency',fontsize=9)
        a3.set_xlim(-3,103)
        a3.set_ylim(0.0,1.)
        #ppl.plot(a2,levs,reliability,linestyle='-',marker='o',color='r')
        if bootstrap:
            a2.errorbar(levs,reliability,yerr=[reliability-low,high-reliability],ecolor='k',fmt='ro')
        a2.plot(levs,levs,linestyle='--',color='k')
        a2.plot(levs,reliability,linestyle='-',marker='o',color='r')
        a2.xaxis.set_ticks(levs)
        a2.yaxis.set_ticks(levs)
        a2.minorticks_on()
        a2.set_ylabel('Observed Relative Frequency')
        a2.set_xlabel('Forecast Probability')
        a2.set_title('Reliability',fontsize=12)
        a2.set_ylim(0,levs[-1])
        a2.set_xlim(0,levs[-1])

    return fig