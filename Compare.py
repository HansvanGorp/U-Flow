# %% imports
# standard libraries
import torch
import argparse

# local
import majority_voting
import overnight_statistics
import printing

# %% loading
# user inputs lists of names to compare against, default is the methods implemented in the manuscript
parser = argparse.ArgumentParser(description='compare results')
parser.add_argument('-names_to_compare',type=str,nargs="+",default=['U-Net_fact','U-Net_drop','Stanford','U-Sleep','U-Flow'])
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
    
# calculate majority voting results such as accuracy
majority_voing_results = majority_voting.get_majority_voting_results(majority_grth,majority_methods)

# calculate overnight statistics
overnight_statistics_grth = overnight_statistics.get_all_overnight_stats(predictions_grth)
overnight_statistics_methods = []
for method_id in range(no_methods):
    overnight_statistics_here = overnight_statistics.get_all_overnight_stats(predictions_methods[method_id])
    overnight_statistics_methods.append(overnight_statistics_here)

# %% calculate the overnight statistics results
overnight_statistics_results = overnight_statistics.get_overnight_statistics_results(overnight_statistics_grth,overnight_statistics_methods)

# %% print the results in tables
row_names = [ 'Accuracy           ',
              'Cohen\'s kappa      ',
              'F1 - Wake          ',
              'F1 - N1            ',
              'F1 - N2            ',
              'F1 - N3            ',
              'F1 - REM           ']

table_name = 'Vote results'
values = majority_voing_results
printing.print_table(table_name, row_names, args.names_to_compare, values, lower=False)

row_names = [ 'TST          [min] ',
              'Sleep Effic.   [%] ',
              'SOL          [min] ',
              'REM latency  [min] ',
              'WASO         [min] ',
              'REM awakenings [-] ',
              'NREM awakenings[-] ',
              'Time in N1   [min] ',
              'Time in N2   [min] ',
              'Time in N3   [min] ',
              'Time in REM  [min] ']

table_name = 'KL divergence'
values = overnight_statistics_results[0]
printing.print_table(table_name, row_names, args.names_to_compare, values)

table_name = 'WD metric'
values = overnight_statistics_results[1]
printing.print_table(table_name, row_names, args.names_to_compare, values)

table_name = 'KS metric'
values = overnight_statistics_results[2]
printing.print_table(table_name, row_names, args.names_to_compare, values)