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
parser.add_argument('-names_to_compare',type=str,nargs="+",default=['U-Net','U-Flow'])
parser.add_argument('-subject_id',type=int,default=34)
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
    
# %% append results into single lists
predictions = predictions_methods + [predictions_grth]
majority_votes = majority_methods + [majority_grth]
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