# %% imports
# standard libs
import numpy as np
import mne

# %% extract label data            
def extract_labels(file,max_epochs,patient_id):
    # get the file as collumns
    with open(file) as f:
        collumns = zip(*[line for line in f])
    
    # extract only the usefull collumns
    scoring_here = []
    for i,collumn in enumerate(collumns):
        if i % 3 == 0:
            scoring = collumn[1:]
            scoring_as_array = np.asarray(scoring,dtype=int)
                         
            # fit all scorings into the same amount
            scoring_as_array_full_length = np.zeros(max_epochs,dtype=int)-1
            scoring_as_array_full_length[0:len(scoring_as_array)] = scoring_as_array
            scoring_as_array_full_length[scoring_as_array_full_length==5] = 4
            scoring_as_array_full_length[scoring_as_array_full_length==7] = -1
            scoring_here.append(scoring_as_array_full_length)
            
    # stack the scorings of the 6 different scorers
    scoring_here = np.vstack(scoring_here)

    # change scorer 2-6 based on cross correlation with scorer 1 (since they are not yet properly synchronized)
    scoring_here_2 = np.zeros_like(scoring_here)-1
    
    # offset of 100
    scoring_here_2[0,100:] = scoring_here[0,:1792-100]
    
    for i in range(1,6):
        lag = np.argmax(np.correlate(scoring_here_2[0,:],scoring_here[i,:],'same')) - 1792//2
        scoring_here_2[i,lag:] = scoring_here[i,:1792-lag]
    
    # remove offset of 100
    scoring_here_3 = np.zeros_like(scoring_here)-1
    scoring_here_3[:,:1792-100] = scoring_here_2[:,100:]
    
    # remove any scoring where one of the scorers did not score, with a special case for subject 35, where one of the scorers is missing
    if patient_id != 35:
        mask = np.any(scoring_here_3==-1,axis=0)
        scoring_here_3[:,mask] = -1
        
    else:
        scoring_here_4 = np.vstack((scoring_here_3[:2],scoring_here_3[2+1:]))
        mask = np.any(scoring_here_4==-1,axis=0)
        scoring_here_3[:,mask] = -1
        
    # remove leading unscored
    lag = mask.argmin()
        
    scoring_here_5 = np.zeros_like(scoring_here)-1
    scoring_here_5[:,:1792-lag] = scoring_here_3[:,lag:]
        
    return scoring_here_5, lag

# %% scaling of PSG using the 95-th percentile
def scale(PSG):
    s = np.percentile(np.abs(PSG),95,axis=1)
    s = np.repeat(np.expand_dims(s,1),PSG.shape[1],axis=1)
    scaled_PSG = np.sign(PSG)*np.log(np.abs(PSG)/s+1)
    return scaled_PSG

# %% extract edf data
def extract_edf(file,max_epochs, lag):
    # read edf
    edf_data = mne.io.read_raw_edf(file,verbose=False)
    
    # extract raw data 
    raw_data = edf_data.get_data()
    channels = edf_data.ch_names
    no_samples = raw_data.shape[1]
    
    # extract all the channels of interest
    C3 = raw_data[[channels.index(i) for i in channels if 'C3' in i][0],:]
    C4 = raw_data[[channels.index(i) for i in channels if 'C4' in i][0],:]
    O1 = raw_data[[channels.index(i) for i in channels if 'O1' in i][0],:]
    O2 = raw_data[[channels.index(i) for i in channels if 'O2' in i][0],:]
    
    LOC = raw_data[[channels.index(i) for i in channels if 'LOC' in i][0],:]
    ROC = raw_data[[channels.index(i) for i in channels if 'ROC' in i][0],:]
    
    EMG = raw_data[[channels.index(i) for i in channels if 'EMG' in i][0],:]
    
    ECG = raw_data[[channels.index(i) for i in channels if 'ECG' in i][0],:]

    data = np.vstack((C3,C4,O1,O2,LOC,ROC,EMG,ECG))
    
    # remove the epochs before start
    start = lag*30*128
    data_windowed = np.zeros((8,max_epochs*30*128))
    data_windowed[:,:(no_samples-start)] = data[:,start:]
    
    return data_windowed
    