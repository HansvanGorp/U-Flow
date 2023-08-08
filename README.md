# Introduction
This code can be used to replicate the results reported in the paper: ["Aleatoric Uncertainty Estimation of Overnight Sleep Statistics Through Posterior Sampling Using Conditional Normalizing Flows"](https://ieeexplore.ieee.org/abstract/document/10096894). 

## Environment
To make use of the environment, first install [Anaconda](https://www.anaconda.com/). In anaconda prompt navigate to the parent directory and run:
```
conda env create -f environment.yml --prefix ./env
```
and then activate the environment:
```
conda activate ./env
```

## unzipping the predictions
The run the comparison and plotting scripts, please first unzip "predictions.zip" in the root directory

## Compare
To recreate the results as listed in the tables of the manuscript, run:
```
python Compare.py
```
Optionally one can specify which method to compare by passing their names as an argument:
```
python Compare.py -names_to_compare= U-Net U-Flow 
```

## plot
To recreate the plots of the manuscript, run:
```
python plot.py
```
Optionally one can specify which method to compare by passing their names as an argument, additonally the subject to plot can also be specified:
```
python plot.py -names_to_compare U-Net U-Flow  -subject_id 34
```

## preprocess
To mimic the preprocessing performed on the IS-RC dataset, first download the dataset from here: [link](https://stanfordmedicine.app.box.com/s/r9e92ygq0erf7hn5re6j51aaggf50jly/folder/53209541138).

Then, prepocessing the dataset can be performed by running:
```
python preprocess.py -data_loc="location//of//the//data"
```
with the data_loc argument pointing towards the location of the raw IS-RC dataset.