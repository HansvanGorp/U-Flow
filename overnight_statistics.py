# %% imports
# standard libraries
import torch
from tqdm import tqdm

# %% caclulate statistics functions
def get_TST(predictions_subject):
    _predictions_subject = predictions_subject.clone()
    _predictions_subject[predictions_subject==0]=-1
    TST = (_predictions_subject!=-1).sum(dim=1)/2
    return TST

def split_into_bouts(predictions_subject):    
    bouts = []
    
    for i in range(predictions_subject.size(0)):
        predictions_here = predictions_subject[i,:]
        bouts_here = torch.tensor_split(predictions_here, torch.where(torch.diff(predictions_here) != 0)[0]+1)
        bouts.append(bouts_here)
    return bouts

def get_awakenings(predictions_subject):
    bouts = split_into_bouts(predictions_subject)
    
    RA = torch.zeros(len(bouts))
    NA = torch.zeros(len(bouts))
    for i,bout in enumerate(bouts):
        for j,b in enumerate(bout):
            if j != 0:
                if bout[j][0] == 0:
                    if bout[j-1][0] == 4:
                        RA[i] += 1
                    else:
                        NA[i] += 1
    return RA, NA
    

def get_TIS(predictions_subject):
    TIS = torch.zeros(predictions_subject.size(0),5)
    for i in range(5):
        TIS[:,i] = (predictions_subject==i).sum(dim=1)/2
    return TIS

def get_all_overnight_stats(predictions):
    no_subjects     = predictions.shape[0]
    overnight_statistics = [[None for i in range(7)] for i in range(no_subjects)]
    
    # loop over subjects
    for subject_id in tqdm(range(no_subjects)):
        predictions_subject = predictions[subject_id,:,:]
        
        #check if predictions is empty (missing one scorer in hold-out test set)
        if torch.any(predictions_subject[:,0]==-1): # if any start of with -1 do a thorough check
            exist = torch.any(predictions_subject!=-1,dim=1)
            predictions_subject = predictions_subject[exist,:]
        
        # calculate stats
        TST   = get_TST(predictions_subject)       # Total Sleep Time
        RA,NA = get_awakenings(predictions_subject)# REM awakenings and NREM awakenings
        TIS   = get_TIS(predictions_subject)       # Time in Stage
        
        # save to list
        overnight_statistics[subject_id][0] = TST
        overnight_statistics[subject_id][1] = TIS[:,1]
        overnight_statistics[subject_id][2] = TIS[:,2]
        overnight_statistics[subject_id][3] = TIS[:,3]
        overnight_statistics[subject_id][4] = TIS[:,4]
        overnight_statistics[subject_id][5] = RA
        overnight_statistics[subject_id][6] = NA
        
    return overnight_statistics

# %% caclutate metrics on those statistics
def calculate_KL(mu1, sigma1, mu2, sigma2):
    if sigma1 < 1e-2:
        sigma1 = 1e-2
            
    if sigma2 < 1e-2:
        sigma2 = 1e-2
    
    
    KL_here = torch.log(torch.ones(1)*sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 1/2
    return KL_here

def get_overnight_statistics_results(overnight_statistics_grth,overnight_statistics_methods):
    no_methods = len(overnight_statistics_methods)
    no_subjects = len(overnight_statistics_grth)
    no_stats = 7
    overnight_statistics_results = torch.zeros(no_stats,no_methods,no_subjects)

    for stat_id in range(no_stats):
        for method_id in range(no_methods):
            for subject_id in range(no_subjects):
                samples_here = overnight_statistics_methods[method_id][subject_id][stat_id]
                samples_grth = overnight_statistics_grth[subject_id][stat_id]
                
                # KL divergence
                mu_here = samples_here.mean()
                st_here = samples_here.std()
                
                mu_grth = samples_grth.mean()
                st_grth = samples_grth.std()
                
                overnight_statistics_results[stat_id, method_id, subject_id] = calculate_KL(mu_grth,st_grth,mu_here,st_here)
                
    return overnight_statistics_results