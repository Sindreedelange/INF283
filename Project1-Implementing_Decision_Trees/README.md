# Implementing Decision Trees

## Project Organization
------------


    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── classes            <- .Py files used in the Notebook:
    │                          ImpurityMeasure - meassures the information gain for a system
    │                           DataCleaning - cleans the data 
    │
    ├── nbs                <- The One Jupyter Notebook To Rule Them All - run the project from here
    │
    │
    ├── reports            <- PDf-file answer to task sheet
------------

## TODO
- .yaml-file for environment set up
- extract methods to seperate .py classes

### DATA
The models in the *.ipynb*-file assumes that the user cleans the data into categorical 
variables only. This can be done using */classes/DataCleaning.py*

## Description

This project assumes that the user has:
- jupyter notebook
- python, with the following packages:
    - numpy
    - pandas <br>
 If this is not the case, well then that it what the '.yaml' point under **TODO** is all about,
 however these can easily be downloaded - i recommend using Conda: https://conda.io/docs/
 - treelib. This can be downloaded directly from the Jupyter Notebook (first cell) 
 
 ## Run the code
 In order to run the code, open the .ipynb file */nbs/Implementing_Decision_Trees_Notebook.ipynb*
 where it is pretty straight forward with comments ment to serve as explanations. 

## Known problems
Pruning

## Disclaimer
If one tries to run **learn()** with pruning one's computer might not like you very much.. 
