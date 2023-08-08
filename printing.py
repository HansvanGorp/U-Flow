# %% imports
# standard libraries
import torch
from scipy.stats import ttest_rel 

# %%
def print_table(table_name, row_names, headers, values, lower = True): 
    # create header
    no_methods = len(headers)
    top = table_name.center(19)+'|'
    for method_id in range(no_methods):
        top += headers[method_id].center(12)
        top += '|'
    
    print('\n\n')
    print(top)
    print('-------------------'+no_methods*'-------------'+'-')
    
    # test for significance
    means = values.mean(dim=-1)
    id_1 = torch.zeros(len(row_names)).type(torch.long)
    id_2 = torch.zeros(len(row_names)).type(torch.long)
    significant = torch.zeros(len(row_names)).type(torch.bool)
    for row_id,row_name in enumerate(row_names):
        means_here = means[row_id,:]
        sorted_ids = means_here.argsort()
        if lower==True:
            id_1[row_id] = sorted_ids[0]
            id_2[row_id] = sorted_ids[1]
        else:
            id_1[row_id] = sorted_ids[-1]
            id_2[row_id] = sorted_ids[-2]
    
        samples_1 = values[row_id,id_1[row_id],:]
        samples_2 = values[row_id,id_2[row_id],:]
        
        res = ttest_rel(samples_1,samples_2)     
        pvalue =  res.pvalue
        
        significant[row_id] = (pvalue<(0.05/len(row_names)))*1.0
        
        
    # fill in table
    for row_id,row_name in enumerate(row_names): 
        print(f"{row_name:<14}",end="")
        for method_id in range(no_methods):
            to_print = values[row_id,method_id,:].mean()
            
            if id_1[row_id] == method_id and significant[row_id]==True:
                print(f"| {to_print:9.3f}* ",end="")
            else:
                print(f"| {to_print:9.3f}  ",end="")
            
            
            
        print("|")