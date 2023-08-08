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

def get_RT(predictions_subject):
    RT = (predictions_subject[0,:]!=-1).sum()/2
    return RT

def get_SE(predictions_subject):
    TST = get_TST(predictions_subject)
    RT  = get_RT( predictions_subject)
    SE = 100 * TST/RT
    return SE

def get_SOL(predictions_subject):
    _predictions_subject = predictions_subject.clone()
    _predictions_subject[predictions_subject==-1] = 0
    
    SOL = torch.argmax((_predictions_subject!=0)*1,dim=1)/2
    return SOL

def get_REMOL(predictions_subject):
    REMOL = torch.argmax((predictions_subject==4)*1,dim=1)/2 - get_SOL(predictions_subject)
    return REMOL

def get_WASO(predictions_subject):
    # total size
    size = (predictions_subject.size(1)/2)
    
    # total sleep time
    TST = get_TST(predictions_subject)
    
    # first wake
    first_wake = get_SOL(predictions_subject)
    
    # last wake
    _predictions_subject = predictions_subject.clone()
    last_wake = get_SOL(_predictions_subject.flip(dims=(1,)))
    
    # WASO
    WASO = size - TST - first_wake - last_wake
    
    return WASO

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
    overnight_statistics = [[None for i in range(12)] for i in range(no_subjects)]
    
    # loop over subjects
    for subject_id in tqdm(range(no_subjects)):
        predictions_subject = predictions[subject_id,:,:]
        
        #check if predictions is empty (missing one scorer in hold-out test set)
        if torch.any(predictions_subject[:,0]==-1): # if any start of with -1 do a thorough check
            exist = torch.any(predictions_subject!=-1,dim=1)
            predictions_subject = predictions_subject[exist,:]
        
        # calculate stats
        TST   = get_TST(predictions_subject)       # Total Sleep Time
        SE    = get_SE(predictions_subject)        # Sleep Efficiency
        SOL   = get_SOL(predictions_subject)       # Sleep Onset Latency
        REMOL = get_REMOL(predictions_subject)     # REM Onset Latency
        WASO  = get_WASO(predictions_subject)      # Wake After Sleep Onset
        RA,NA = get_awakenings(predictions_subject)# REM awakenings and NREM awakenings
        TIS   = get_TIS(predictions_subject)       # Time in Stage
        RT    = get_RT(predictions_subject)        # Recording Time
        
        # save to list
        overnight_statistics[subject_id][0] = TST
        overnight_statistics[subject_id][1] = SE
        overnight_statistics[subject_id][2] = SOL
        overnight_statistics[subject_id][3] = REMOL
        overnight_statistics[subject_id][4] = WASO
        overnight_statistics[subject_id][5] = RA
        overnight_statistics[subject_id][6] = NA
        overnight_statistics[subject_id][7] = TIS[:,1]
        overnight_statistics[subject_id][8] = TIS[:,2]
        overnight_statistics[subject_id][9] = TIS[:,3]
        overnight_statistics[subject_id][10]= TIS[:,4]
        overnight_statistics[subject_id][11]= RT
        
    return overnight_statistics

def get_overnight_stats_one_subject(predictions, subject_id):
    overnight_statistics = [None for i in range(12)]

    predictions_subject = predictions[subject_id,:,:]
    
    #check if predictions is empty (missing one scorer in hold-out test set)
    if torch.any(predictions_subject[:,0]==-1): # if any start of with -1 do a thorough check
        exist = torch.any(predictions_subject!=-1,dim=1)
        predictions_subject = predictions_subject[exist,:]
    
    # calculate stats
    TST   = get_TST(predictions_subject)       # Total Sleep Time
    SE    = get_SE(predictions_subject)        # Sleep Efficiency
    SOL   = get_SOL(predictions_subject)       # Sleep Onset Latency
    REMOL = get_REMOL(predictions_subject)     # REM Onset Latency
    WASO  = get_WASO(predictions_subject)      # Wake After Sleep Onset
    RA,NA = get_awakenings(predictions_subject)# REM awakenings and NREM awakenings
    TIS   = get_TIS(predictions_subject)       # Time in Stage
    RT    = get_RT(predictions_subject)        # Recording Time
    
    # save to list
    overnight_statistics[0] = TST
    overnight_statistics[1] = SE
    overnight_statistics[2] = SOL
    overnight_statistics[3] = REMOL
    overnight_statistics[4] = WASO
    overnight_statistics[5] = RA
    overnight_statistics[6] = NA
    overnight_statistics[7] = TIS[:,1]
    overnight_statistics[8] = TIS[:,2]
    overnight_statistics[9] = TIS[:,3]
    overnight_statistics[10]= TIS[:,4]
    overnight_statistics[11]= RT
        
    return overnight_statistics

# %% caclutate metrics on those statistics
def calculate_KL(mu1, sigma1, mu2, sigma2):
    if sigma1 < 1e-2:
        sigma1 = 1e-2
            
    if sigma2 < 1e-2:
        sigma2 = 1e-2
    
    
    KL_here = torch.log(torch.ones(1)*sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 1/2
    return KL_here

def convert_2_cdf(samples):
    x_min = -208/2
    x_max = 1792/2
    no_bins = 2000
    
    bins = torch.arange(no_bins)/2 + x_min
    
    hist = torch.histc(samples, bins = no_bins, min=x_min, max=x_max)
    CDF = hist.cumsum(dim=0)
    CDF = CDF/CDF[-1]
    
    return bins, CDF

def get_overnight_statistics_results(overnight_statistics_grth,overnight_statistics_methods):
    no_methods = len(overnight_statistics_methods)
    no_subjects = len(overnight_statistics_grth)
    no_stats = 11
    overnight_statistics_results = torch.zeros(3,no_stats,no_methods,no_subjects)

    for stat_id in range(no_stats):
        for method_id in range(no_methods):
            for subject_id in range(no_subjects):
                samples_here = overnight_statistics_methods[method_id][subject_id][stat_id]
                samples_grth = overnight_statistics_grth[subject_id][stat_id]
                
                if stat_id == 1:
                    # Sleep Efficiency, bin_multiplier = Recording_time / 100
                    bin_multiplier = 100 / overnight_statistics_grth[subject_id][11]
                else:
                    # normal stats, bin_multiplier = 1
                    bin_multiplier = 1

                _, CDF_here = convert_2_cdf(samples_here/bin_multiplier)
                _, CDF_grth = convert_2_cdf(samples_grth/bin_multiplier)
                
                diff = torch.abs(CDF_here-CDF_grth)
                
                # KL divergence
                mu_here = samples_here.mean()
                st_here = samples_here.std()
                
                mu_grth = samples_grth.mean()
                st_grth = samples_grth.std()
                
                overnight_statistics_results[0,stat_id, method_id, subject_id] = calculate_KL(mu_grth,st_grth,mu_here,st_here)
                
                # Wasserstein metric
                bin_width = 0.5 * bin_multiplier
                overnight_statistics_results[1,stat_id, method_id, subject_id] = diff.sum()*bin_width # we have to multiply by bin_width as WD depends on x-axis scale
                
                # Kolmogorov-Smirnov metric
                overnight_statistics_results[2,stat_id, method_id, subject_id] = diff.max()

                
    return overnight_statistics_results