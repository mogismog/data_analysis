# -*- coding: utf-8 -*-
"""
@author: falvarez
"""
#!/usr/bin/env python

import numpy as np
import multiprocessing as mp
import os
import matplotlib.pyplot as plt

try:
    import prettyplotlib as ppl
except:
    ppl_flag = False
else:
    ppl_flag = True


def boxcox_yeo(predictor,power_transforms):
    """
    
        Function to do Box-Cox tranforms on a 1-D NumPy array.
        If non-positive values are found, we use the Yeo Johnson (2000)
        family of transforms, which keep almost all of the awesome qualities
        found from Box-Cox trasforms.
        
        Input:
            predictor - 1-D NumPy Array
            power_transforms - some list of power transforms
        Output:
            transformed_preds - n-D NumPy array, n = len(power_transforms)
    """
    
    transformed_preds = np.zeros((len(power_transforms),predictor.shape[0]))
    if predictor.min() <= 0:
        for pwr in xrange(len(power_transforms)):
            
            # --- First, transforms for non-negative value
            if power_transforms[pwr] == 0:
                msk = (predictor >=0)
                transformed_preds[pwr,msk] = np.log(predictor[msk]+1.)       
            if power_transforms[pwr] != 0:
                msk = (predictor >=0)
                transformed_preds[pwr,msk] =  ((predictor[msk]+1.)**(power_transforms[pwr]) - 1.)/power_transforms[pwr]        
            
            # --- Now, transforms for negative value
            if power_transforms[pwr] == 2:
                msk = (predictor < 0)
                transformed_preds[pwr,msk] =  -np.log((predictor[msk]*-1.)+1.)                 
            if power_transforms[pwr] != 2:
                msk = (predictor < 0)
                transformed_preds[pwr,msk] = -1.*(((predictor[msk]*-1.)+1.)**(2.-power_transforms[pwr]) - 1.)/(2.-power_transforms[pwr])
    else:
        for pwr in xrange(len(power_transforms)):
            
            # --- First, transforms for non-negative value
            if power_transforms[pwr] == 0:
                transformed_preds[pwr,:] = np.log(predictor[:])       
            if power_transforms[pwr] != 0:
                transformed_preds[pwr,:] =  ((predictor[:])**(power_transforms[pwr]) - 1.)/power_transforms[pwr]      
    
    return transformed_preds
    
def relative_frequency(fig,predictor,predictand,var_name,**kwargs):
    """
        A function that generates reliability diagrams.
        
        Input:
        
        fig - Some matplotlib figure object
        predictor - a 1-D NumPy array of some values
        predictand - a 1-D NumPy array of binary values
        var_name - String of the name of the predictor
        
        Optional Arguments:

    """
    
    power_transforms = kwargs.get('power_transforms',[-3.,-2.,-1.,-.5,0.,.5,1.,2.,3.])
    n_bins = kwargs.get('n_bins',50)
    n_cols = kwargs.get('n_cols',3)
    n_rows = kwargs.get('n_rows',3)
    

    # --- Set up indices, arrays and masks for relative frequency plots
    bin_idxs = np.floor(np.linspace(1,predictor.shape[0],n_bins+1))
    relative_frequency = np.zeros((n_bins))
    mean_samps = np.zeros((n_bins))
    dummy_array_for_plot = np.zeros(n_bins)

    # --- Combine forecast/verif arrays into a rec array to make this easier
    rec_arrays = np.rec.fromarrays([predictor,predictand])
    rec_arrays.sort(order='f0')
    
    X = rec_arrays.f0
    X = boxcox_yeo(X,power_transforms)
    
    for pwr in xrange(len(power_transforms)):
        
        for n_bin in xrange(bin_idxs.shape[0]-1):
            all_samples = X[pwr,int(bin_idxs[n_bin]):int(bin_idxs[n_bin+1])]
            n_samps = all_samples.shape[0]
            tor_count = np.sum(rec_arrays.f1[int(bin_idxs[n_bin]):int(bin_idxs[n_bin+1])])
            Prob = float(tor_count)/float(n_samps) # --- Probability of tornado occurring in range of var
            Prob_over_1minusProb = Prob/(1-Prob) # --- P/(1-P)
            relative_frequency[n_bin] = Prob_over_1minusProb#*100.
            mean_samps[n_bin] = all_samples.mean()
    
        
        ax = fig.add_subplot(n_rows,n_cols,pwr+1)
        ppl.fill_between(mean_samps,relative_frequency,dummy_array_for_plot,color='green',alpha=.5)
        if power_transforms[pwr] == 0:
            ax.set_xlabel('log({})'.format(var_name))
        else:
            ax.set_xlabel('{}**{}'.format(var_name,power_transforms[pwr]))
        #ax.set_ylabel('Relative Frequency, P/(1-P)')
        ax.set_xlim(mean_samps[0]-mean_samps[1],mean_samps[-1])
        ax.set_ylim(0,relative_frequency.max()+.01)
        #fig_title = "x**{}".format(power_transforms[pwr])
        #fig.text(.5,.965,fig_title,ha='center',va='center',fontsize=14)
    plt.savefig('power_transform_{}.png'.format(var_name),dpi=300)
    plt.clf()
    plt.close()
            
