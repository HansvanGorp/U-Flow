# %% imports
# standard libraries
import torch
import argparse
import matplotlib.pyplot as plt

# local
import majority_voting
import plot_functions
import overnight_statistics

# %% loading
# user inputs lists of names to compare against, default is the methods implemented in the manuscript
parser = argparse.ArgumentParser(description='compare results')
parser.add_argument('-names_to_compare',type=str,nargs="+",default=['U-Net_fact','U-Net_drop','U-Flow'])
parser.add_argument('-subject_id',type=int,default=19)
args = parser.parse_args()

# load the predictions
predictions_grth = torch.load("predictions/ground_truth.tar")
predictions_methods = []
no_methods = len(args.names_to_compare)
for method_id in range(no_methods):
    predictions_here = torch.load(f"predictions/{args.names_to_compare[method_id]}.tar")
    predictions_methods.append(predictions_here)
    
# calculate majority votes
majority_grth = majority_voting.get_majority_vote_human_panel(predictions_grth)
majority_methods = []
for method_id in range(no_methods):
    majority_here = majority_voting.get_majority_vote(predictions_methods[method_id])
    majority_methods.append(majority_here)


# calculate overnight statistics
overnight_statistics_grth = overnight_statistics.get_overnight_stats_one_subject(predictions_grth,args.subject_id)
overnight_statistics_methods = []
for method_id in range(no_methods):
    overnight_statistics_here = overnight_statistics.get_overnight_stats_one_subject(predictions_methods[method_id],args.subject_id)
    overnight_statistics_methods.append(overnight_statistics_here)
    
# %% append results into single lists
predictions = predictions_methods + [predictions_grth]
majority_votes = majority_methods + [majority_grth]
overnight_statistics = overnight_statistics_methods + [overnight_statistics_grth]
names = args.names_to_compare + ["ground truth"]

# %% plot the hypnograms
mask = (majority_grth[args.subject_id,:]!=-1)
min_t = 0
max_t = torch.ceil(((mask*1).argmin())/60)/2
no_methods = len(names)
plt.figure(figsize=(5*no_methods,7.5))

for j,(prediction,majority_vote,name) in enumerate(zip(predictions,majority_votes,names)):        
    for i in range(6):
        plt.subplot(8,no_methods,j+i*no_methods+1)
        plot_functions.plot_hypnogram(prediction[args.subject_id,i,:])
        plt.ylabel('')
        plt.grid()
        
        
        if i == 0:
            plt.title(f"hypnograms for {name}")
        if i != 5:
              plt.xticks([0,2,4,6,8],[])
        
        plt.xlim(min_t,max_t)
        
    # majority vote
    plt.subplot(8,no_methods,j+7*no_methods+1)
    plot_functions.plot_hypnogram(majority_vote[args.subject_id,:])
    plt.ylabel('')
    plt.xlim(min_t,max_t)
    plt.grid()
    plt.xlabel("hours of sleep")
    plt.title(f"majority vote for {name}")
    
plt.savefig("figures//hypnograms.png",dpi=300,bbox_inches='tight')
plt.close()

# %% eCDF of overnight statistics
stat_names = ['Total Sleep Time', 
              'Sleep Efficiency',
              'Sleep Onset Latency', 
              'REM Latency', 
              'WASO', 
              'REM awakenings ',
              'NREM awakenings',
              'Time in N1',
              'Time in N2',
              'Time in N3',
              'Time in REM']

for i, stat_name in enumerate(stat_names):
    plt.figure()
    min_x = []
    max_x = []
    for j,stats in enumerate(overnight_statistics):  
        samples = stats[i]
        
        if i == 1:
            # Sleep Efficiency, bin_multiplier = Recording_time / 100
            bin_multiplier = 100 / stats[11]
        else:
            # normal stats, bin_multiplier = 1
            bin_multiplier = 1

        min_x_here, max_x_here = plot_functions.plot_eCDF(samples, bin_multiplier)
        
        min_x.append(min_x_here)
        max_x.append(max_x_here)
        
    # title
    if j not in [4,5]:
        plt.xlabel('time [min]')
    else:
        plt.xlabel('awakenings [-]')    
    
    plt.ylabel('cumulative probability')
    
    plt.title(stat_name)
    
    # min and max
    x_min2 = min(min_x)*0.95
    x_max2 = max(max_x)*1.05
    
    if x_min2 <= 5:
        x_min2 = -1
        
    plt.xlim(x_min2,x_max2)
    
    plt.legend(names)
    plt.grid()
    plt.savefig(f"figures//{stat_name}.png",dpi=300,bbox_inches='tight')
    plt.close()
