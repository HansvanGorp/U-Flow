# %% imports
# standard libraries
import torch
import matplotlib.pyplot as plt
import numpy as np

# local
from overnight_statistics import convert_2_cdf

# %% plot a single hypnogram
def plot_hypnogram(y):
    y = y.float()
    
    # create time variable
    time = (torch.arange(y.size(0))*30)/3600
    
    # repeat with interleaving
    time = time.repeat_interleave(2)
    y = y.repeat_interleave(2)
    
    # remove first 0
    time2 = time.clone()
    time[0:y.size(0)-1] = time2[1:]
    
    # add offset to the last one
    time[-1] += (30/3600)
    
    # rearange order
    y2 = torch.zeros_like(y)
    y2[y==-1] = 4
    y2[y==0]  = 4
    y2[y==1]  = 2
    y2[y==2]  = 1
    y2[y==3]  = 0
    y2[y==4]  = 3
    
    # create REM line
    y2_R = y2.clone()
    t_R = time.clone()
    
    y2_R[y2!=3] = np.nan
    t_R[y2!=3] = np.nan
    
    # plot
    plt.plot(time,y2,c='k')
    plt.plot(t_R,y2_R,c='r',linewidth=4)
    plt.grid()
    plt.yticks([0,1,2,3,4],['N3','N2','N1','REM','Wake'])
    plt.ylim(-0.2,4.2)
    
# %% plot an eCDF
def plot_eCDF(samples, bin_multiplier):
    bins,CDF = convert_2_cdf(samples / bin_multiplier)
    bins = bins * bin_multiplier
    
    # repeat with interleaving
    bins = bins.repeat_interleave(2)
    CDF = CDF.repeat_interleave(2)
    
    # remove first 0
    bins2 = bins.clone()
    bins[0:3999] = bins2[1:]
    
    # add offset to the last one
    bins[-1] += 0.5
    
    # plot it
    plt.plot(bins, CDF)
    
    # get the min and max
    CDF_min = ((CDF>0.05)*1).argmax()
    CDF_max = ((CDF<(1-0.05))*-1).argmax()
    
    min_x = bins[CDF_min]
    max_x = bins[CDF_max]
    
    return min_x, max_x

