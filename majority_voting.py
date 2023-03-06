# %% imports
# standard libraries
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

# %% majority voting results
def get_majority_voting_results(majority_grth,majority_methods):
    no_methods = len(majority_methods)
    no_subjects = majority_grth.size(0)
    majority_voing_results = torch.zeros(2,no_methods,no_subjects)
    masks = (majority_grth[:,:]!=-1)

    # loop over methods and subjects
    for method_id in range(no_methods):
        majority_here = majority_methods[method_id]
        for subject_id in range(no_subjects):     
            # get rid of unscored segments
            mask = masks[subject_id,:]
            majority_grth_masked = majority_grth[subject_id,mask]
            majority_here_masked = majority_here[subject_id,mask]
            
            # acc
            majority_voing_results[0,method_id,subject_id] = 100*(majority_grth_masked==majority_here_masked).sum()/majority_grth_masked.size(0)
            
            # kappa
            majority_voing_results[1,method_id,subject_id] = cohen_kappa_score(majority_grth_masked,majority_here_masked)
            
    return majority_voing_results

# %% get majority vote
def get_majority_vote(predictions):
    votes = torch.zeros(predictions.size(0),6,predictions.size(2))
    
    #go over stages
    for i in range(6):
        votes[:,i,:] = (predictions == (i-1)).sum(dim=1)
    
    # naive majority vote with a preference in order: Wake, N1, N2, N3, REM
    majority_vote = votes.argmax(dim=1)-1
    
    return majority_vote
    
# %% majority vote human panel
def get_majority_vote_human_panel(predictions):
    predictions = predictions.long()
    
    # allocate an array for the majority vote of all nights
    majority_vote_all = torch.zeros(predictions.size(0),predictions.size(2))
    
    # loop over each night
    for n in range(predictions.size(0)):
        # loop over each scorer to geth their soft-agreement for this night
        soft_agreements = torch.zeros(predictions.size(1))
        for s in range(predictions.size(1)):
            # group
            group = np.ones((predictions.size(1),),bool)
            group[s] = False
            
            # get the prediction of this scorer
            prediction_here = predictions[n,s,:]
            
            # get the prediction of the remaining group (N-1) scorers
            prediction_rest = predictions[n,group,:]
            
            # calculate soft agreement with majority vote of remaining group
            T = torch.arange(predictions.size(2))
            z = torch.zeros(6,predictions.size(2))
            for i in range(6):
                z[i,:] = (prediction_rest == (i-1)).sum(dim=0)
            z_max,_ = z.max(dim=0)
            z = z/z_max
            
            soft_agreements[s] = z[prediction_here+1, T].mean()
            
        # get the best scoring
        best_scorer_index = soft_agreements.argmax()
        best_scoring = predictions[n,best_scorer_index,:]
        
        # get the majority vote using all scorers
        votes = torch.zeros(6,predictions.size(2))
        
        #go over stages
        for i in range(6):
            votes[i,:] = (predictions[n,:,:] == (i-1)).sum(dim=0)
        
        # majority vote
        majority_vote = votes.argmax(dim=0)-1
        
        # everywhere there is a tie, follow the most reliable scorer
        max_votes,_ = votes.max(dim=0)
        ties = (votes == max_votes.unsqueeze(0).repeat(6,1))
        ties = ties.sum(dim=0)>1
        majority_vote[ties] = best_scoring[ties]
        
        # save this night
        majority_vote_all[n,:] = majority_vote
        
    return majority_vote_all
