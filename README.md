# Description

Fine-tune ResNet variants on the speech commands dataset.
The ``params.yaml`` config file constains parameters to 
train with MFCC features using ResNext50 as the backbone.

This configuration achieves ~90.9% accuracy on the test set.

Created by Sebastian Parkitny Dec 2022.

# Requirements

At minimum a machine with CUDA 11.x installed on the system.

# Installation

Run:

```bash
conda env create -f environment.yml
conda activate lc
```

# Training


Run:

```python
python -m src.train
```

NOTE: You'll need ~8GB VRAM to run with this config, if you run out of memory,
try lowering ``batch_size`` in ``params.yaml``

# Evaluation

Run:

```python
python -m src.eval
```

With the checkpoint: ``lightning_logs/version_0/checkpoints/lc-challenge-epoch13-val_acc0.91.ckpt``

Alternatively, use the ``-c`` argument to specify a path to another checkpoint.

# Visualisation

Run: 

```python
jupyter-notebook ./exploration
```

Then open the data-exploration notebook. Run each cell one at a time.
Alternatively, the output is in exploration/data-exploration.html

# Additional Info

The min and max values of the MFCC spectrograms for the training set
are calculated by running:

```python
python -m src.data
```

These were computed once then added into ``params.yaml`` for use during
training.
