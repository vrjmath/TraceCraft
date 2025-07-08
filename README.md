# TraceCraft

[[Paper]](https://asplos26-spring.hotcrp.com/doc/asplos26-spring-paper79.pdf)

## Installation

```bash
conda create -n LayerDAG python=3.10 -y
conda activate LayerDAG
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c conda-forge cudatoolkit=11.6
conda clean --all -y
pip install dgl==1.1.0+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
pip install tqdm einops wandb pydantic pandas
pip install numpy==1.26.3
```

## Train

To train a LayerDAG model,

```bash
python train.py --config_file configs/LayerDAG/tpu_tile.yaml
```

The trained model checkpoint will be saved to a file `model_tpu_tile_{time_stamp}.pth`.

## Sample

For sampling and evaluation,

```bash
python sample.py --model_path X
```

where `X` is the file `model_tpu_tile_{time_stamp}.pth` saved above.

## Frequently Asked Questions

### Q1: libcusparse.so

**An error occurs that the program cannot find `libcusparse.so`**, e.g., OSError: libcusparse.so.11: cannot open shared object file: No such file or directory.

To search for the location of it on linux,

```bash
find /path/to/directory -name libcusparse.so.11 -exec realpath {} \;
```

where `/path/to/directory` is the directory you want to search. Assume that the search returns `/home/miniconda3/envs/LayerDAG/lib/libcusparse.so.11`. Then you need to manually specify the environment variable as follows.

```bash
export LD_LIBRARY_PATH=/home/miniconda3/envs/LayerDAG/lib:$LD_LIBRARY_PATH
```

## Citation


