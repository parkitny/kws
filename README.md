# Description

Fine-tune ResNet variants on the speech commands dataset.
The ``params.yaml`` config file constains parameters to 
train with MFCC features using ResNext50 as the backbone.

This configuration achieves ~91.5% accuracy on the test set.

The best checkpoint I've produced so far is:

``lightning_logs/version_1/checkpoints/lc-challenge-epoch10-val_acc0.92.ckpt``

Not all training runs are the same, sometimes re-running will
give slightly better / worse performance.

The metrics reported at the very end of train are not for the
best checkpoint. Once training is done, to get the metrics
for the best checkpoint - please follow the steps under the
``Evaluation`` section.

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

By default, this will find the checkpoint with best ``val_acc`` in the last version in
``lightning_logs``. For the zipped version of this project, it will use this checkpoint: 
``lightning_logs/version_1/checkpoints/lc-challenge-epoch10-val_acc0.92.ckpt``

Alternatively, use the ``-c`` argument to specify a path to another checkpoint.

### Confusion Matrix for the above checkpoint
![plot](./confmat.jpg)

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
