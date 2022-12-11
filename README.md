# Description
Fine-tune ResNet variants on the speech commands dataset.
The ``params.yaml`` config file constains parameters to 
train with MFCC features using ResNext50 as the backbone.

This configuration achieves ~90.9% accuracy on the test set.

# Requirements
At minimum a machine with CUDA 11.x installed on the system.

# Installation
$ conda env create -f environment.yml
$ conda activate lc

# Training
$ python -m src.train

# Evaluation
$ python -m src.eval

# Visualisation
$ jupyter-notebook ./exploration

Then open the data-exploration notebook. Run each cell one at a time.
