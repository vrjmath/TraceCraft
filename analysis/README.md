# TraceCraft Experiments

This folder allows user to reproduce the TraceCraft paper results.

## Requirements

- Python: `3.10.18`
- Conda: `24.3.0` or later

We strongly recommend using a `conda` environment to manage dependencies.

```bash
conda create -n tracecraft-env python=3.10 -y
conda activate tracecraft-env
conda install -c conda-forge networkx scipy numpy pytorch -y
```

On your local machine, create a folder called `analysis/report` and place the dataset files inside the `analysis/dataset` folder.

Then you can run:

```bash
python main.py
```


