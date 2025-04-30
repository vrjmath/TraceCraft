from .layer_dag import *
from .general import DAGDataset
from .tpu_tile import get_tpu_tile
from .traces import get_traces

def load_dataset(dataset_name):
    if dataset_name == 'tpu_tile':
        return get_tpu_tile()
    elif dataset_name == "traces":
        return get_traces()
    else:
        return NotImplementedError
