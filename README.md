Generalised Interpretable Shapelets for Irregular Time-Series
==================================================================
[<a href="">arXiv</a>]

<p align="center">
<img align="middle" src="./paper/images/new_pendigits.png" width="666" />
</p>

A generalised approach to _the shapelet method_ used in time-series classification. This code provides an extension of the shapelet method through its ability to handle
1. Irregular (messy) time-series data.
2. Differentiably optimized shapelet lengths.
3. Interpretable regularisation term to force the 
4. Ability to handle arbitrary shapelet-path discrepancy functions (not just L2) provided we can back-propagate through them.

----
### Downloading the data
The scripts for downloading the data can be found in ther [`get_data`](./get_data) folder. 

### Reproducing experiments
Everything to reproduce the experiments of the paper can be found in the [`experiments`](./experiments) folder. In principle at least, all that needs to be run (assuming the data is downloaded) is:
+ ``python experiments/uea.py``
+ ``python experiments/speech_commands.py``

However due to the high memory cost, we do not advise attempting this in one go and instead suggest using the scripts as a guide to understand how experiments are run, and only running a subset of experiments that are of most interest.

### Results
A table compiling all the accuracy results that are given in the paper. 
<p align="center">
<img align="middle" src="./paper/images/results_table_full.png" width="666" />
</p>
The first 14 MFC coefficients for an audio recording from the Speech Commands dataset, along with the learnt shapelet, and the difference between them. 
<p align="center">
<img align="middle" src="./paper/images/new_speech_commands_heatmap.png" width="666" />
</p>
For further analysis and graphics, check the full paper. 

### Citation
