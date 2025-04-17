import torch
from collections import OrderedDict
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    NodeType,
    CollectiveCommType,
    GlobalMetadata
)
from chakra.src.third_party.utils.protolib import openFileRd as open_file_rd
from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
from torch.utils.data import Dataset, Subset
import os
import json
from tqdm import tqdm
import math
import copy
import pdb

class ChakraNodeToDAGLMNode:
    def __init__(self, chakra_node):
        self.chakra_node = chakra_node
        
    def get_cat_feat(self):
        node_type_map_cat = {
            NodeType.INVALID_NODE: 0,
            NodeType.METADATA_NODE: 1,
            NodeType.MEM_LOAD_NODE: 2,
            NodeType.MEM_STORE_NODE: 3,
            NodeType.COMP_NODE: 4,
            NodeType.COMM_SEND_NODE: 5,
            NodeType.COMM_RECV_NODE: 6,
            NodeType.COMM_COLL_NODE: 7
        }
        coll_type_map_cat = {
            CollectiveCommType.ALL_REDUCE: 0,
            CollectiveCommType.REDUCE: 1,
            CollectiveCommType.ALL_GATHER: 2,
            CollectiveCommType.GATHER: 3,
            CollectiveCommType.SCATTER: 4,
            CollectiveCommType.BROADCAST: 5,
            CollectiveCommType.ALL_TO_ALL: 6,
            CollectiveCommType.REDUCE_SCATTER: 7,
            CollectiveCommType.REDUCE_SCATTER_BLOCK: 8,
            CollectiveCommType.BARRIER: 9
        }
        ret = node_type_map_cat[self.chakra_node.type]
        if self.chakra_node.type == NodeType.COMM_COLL_NODE:
            for attr in self.chakra_node.attr:
                if attr.name == "comm_type":
                    ret += attr.int64_val
                    break
        return int(ret)
    
    def get_dense_feat(self):
        feats = [
            ('y_tensor_size', -1),
            ('num_ops', -1),
            ('tensor_size', -1),
            ('comm_size', -1),
            ('comm_group', -1),
            ('comm_src', -1),
            ('comm_dst', -1),
            ('comm_tag', -1)
        ]
        feat_type = {
            'y_tensor_size': 'uint64_val',
            'num_ops': 'uint64_val',
            'tensor_size': 'uint64_val',
            'comm_size': 'uint64_val',
            'comm_group': 'int32_val',
            'comm_src': 'uint32_val',
            'comm_dst': 'uint32_val',
            'comm_tag': 'uint32_val'
        }
        norm_funcs = {
            # 'num_ops': lambda x: math.log10(x+1),
            # 'tensor_size': lambda x: math.log10(x+1),
            # 'comm_size': lambda x: math.log10(x+1),
            'y_tensor_size': lambda x: x,
            'num_ops': lambda x: x,
            'tensor_size': lambda x: x,
            'comm_size': lambda x: x,
            'comm_group': lambda x: x,
            'comm_src': lambda x: x,
            'comm_dst': lambda x: x,
            'comm_tag': lambda x: x
        }
        non_dense_feats = {'comm_type'}
        feats = OrderedDict(feats)
        for attr in self.chakra_node.attr:
            if attr.name not in feats:
                assert attr.name in non_dense_feats
                continue
            assert attr.name in feats, attr.name
            assert attr.name in feat_type, attr.name
            norm_func = norm_funcs[attr.name]
            if feat_type[attr.name] == 'uint64_val':
                feats[attr.name] = norm_func(attr.uint64_val)
            elif feat_type[attr.name] == 'uint32_val':
                feats[attr.name] = norm_func(attr.uint32_val)
            elif feat_type[attr.name] == 'int32_val':
                feats[attr.name] = norm_func(attr.int32_val)
        # return list(feats.values())
        # pdb.set_trace()
        return [feats['num_ops'], feats['tensor_size'], feats['comm_size'], feats['comm_group']]
    
    def process(self, chakra_node=None):
        if chakra_node is not None:
            self.chakra_node = chakra_node
        assert self.chakra_node is not None
        cat_feat = self.get_cat_feat()
        dense_feat = self.get_dense_feat()
        cat_feat = torch.tensor(cat_feat)
        dense_feat = torch.tensor(dense_feat)
        return cat_feat, dense_feat

class ChakraToDAGLMGraph:
    def __init__(self, chakra_graph):
        if isinstance(chakra_graph, str):
            chakra_graph = open_file_rd(chakra_graph)
        self.chakra_graph = chakra_graph
        self.node_converter = ChakraNodeToDAGLMNode(None)
        
    def process(self):
        edges = list()
        node_id_map_shrinked_id = dict()
        cat_feats = list()
        dense_feats = list()
        
        node = ChakraNode()
        gm = GlobalMetadata()
        decode_message(self.chakra_graph, gm)
        shrinked_id = 0
        while decode_message(self.chakra_graph, node):
            node_id_map_shrinked_id[node.id] = shrinked_id
            for data_dep_id in node.data_deps:
                edges.append((data_dep_id, node.id))
            for ctrl_dep_id in node.ctrl_deps:
                edges.append((ctrl_dep_id, node.id))
                pdb.set_trace()
            cat_feat, dense_feat = self.node_converter.process(node)
            cat_feats.append(cat_feat)
            dense_feats.append(dense_feat)
            shrinked_id += 1
        cat_feats = torch.stack(cat_feats).reshape((-1, 1))
        dense_feats = torch.stack(dense_feats)
        
        shrinked_edges = list()
        for from_, to_ in edges:
            shrinked_from = node_id_map_shrinked_id[from_]
            shrinked_to = node_id_map_shrinked_id[to_]
            shrinked_edges.append((shrinked_from, shrinked_to))
        
        adj = torch.tensor(shrinked_edges)
        src, dst = adj[:, 0], adj[:, 1]
        
        in_deg = torch.zeros((len(cat_feats), 1))
        for i in dst:
            in_deg[i] += 1
        return cat_feats, dense_feats, src, dst, in_deg

class ChakraDataset(Dataset):
    def __init__(self, chakra_et_root, runtimes, only_sym=False, preload=True, get_in_deg=True):
        if isinstance(runtimes, str):
            runtimes = [runtimes]
        self.chakra_files = list()
        for file in os.listdir(chakra_et_root):
            if not file.endswith(".et"):
                continue
            if only_sym:
                if not file.endswith(".0.et"):
                    continue
            self.chakra_files.append(os.path.join(chakra_et_root, file))
            
        self.runtime = self.read_runtime(runtimes)
        
        assert len(self.runtime) == len(self.chakra_files)
        
        self.num_categories = -1
        self.num_real_feat = -1
        self.num_graph_feat = -1
        self.max_in_deg = -1
        self.min_num_nodes = 1e100
        self.max_num_nodes = -1
        self.get_in_deg=get_in_deg
        
        self.preload = preload
        self.preloaded_data = None
        
        assert self.preload
        if self.preload:
            self.preload = False
            self.preloaded_data = list()
            for i in tqdm(range(len(self))):
                data = self[i]
                self.preloaded_data.append(data)
                self.max_in_deg = max(self.max_in_deg, int(max(data[4])))
                num_nodes = data[0].shape[0]
                self.min_num_nodes = min(self.min_num_nodes, num_nodes)
                self.max_num_nodes = max(self.max_num_nodes, num_nodes)
            self.num_categories = [17]
            # self.num_real_feat = self.preloaded_data[0][1].shape[1]
            self.num_real_feat = 1
            self.num_graph_feat = self.preloaded_data[0][5].shape[0]
            self.num_cat_feat = 1
            self.preload = True
            # self.norm_data()
            
    def norm_data(self):
        assert self.preload
        def _runtime_norm(runtimes):
            runtimes = copy.deepcopy(runtimes)
            maxx, minn = None, None
            for runtime in runtimes:
                runtime = torch.log10(runtime+1)
                if maxx is None:
                    assert minn is None
                    maxx = runtime
                    minn = runtime
                    continue
                maxx = torch.max(runtime, maxx)
                minn = torch.min(runtime, minn)
            for i, runtime in enumerate(runtimes):
                runtimes[i] = (torch.log10(runtime+1)-minn) / (maxx-minn) - 0.5
            return runtimes
        self.runtime = _runtime_norm(self.runtime)
        
        # gather
        maxx, minn = None, None
        for i, data in enumerate(self.preloaded_data):
            dense_feats = data[1]
            mask = dense_feats == -1
            dense_feats.masked_fill_(mask, -1e30)
            this_max = torch.max(dense_feats, dim=0).values
            dense_feats.masked_fill_(mask, 1e30)
            this_min = torch.min(dense_feats, dim=0).values
            dense_feats.masked_fill_(mask, -1)
            if maxx is None:
                assert minn is None
                maxx = this_max
                minn = this_min
                continue
            maxx = torch.max(this_max, maxx)
            minn = torch.min(this_min, minn)
        # norm
        maxx, minn = maxx.unsqueeze(0), minn.unsqueeze(0)
        for i, data in enumerate(self.preloaded_data):
            cat_feats, dense_feats, src, dst, in_deg, _ = data
            mask = dense_feats == -1
            dense_feats = (dense_feats - minn) / (maxx-minn)
            rand = torch.rand(dense_feats.shape)
            dense_feats[mask] = rand[mask]
            dense_feats -= 0.5
            runtime = self.runtime[i]
            self.preloaded_data[i] = cat_feats, dense_feats, src, dst, in_deg, runtime.float()
            
    def __len__(self):
        return len(self.chakra_files)
    
    def __getitem__(self, index):
        if self.preload:
            cat_feats, dense_feats, src, dst, in_deg, runtime = self.preloaded_data[index]
            if self.get_in_deg:
                return cat_feats, dense_feats, src, dst, in_deg, runtime
            else:
                return cat_feats, dense_feats, src, dst, runtime
        chakra_file = self.chakra_files[index]
        cat_feats, dense_feats, src, dst, in_deg = ChakraToDAGLMGraph(chakra_file).process()
        runtime = self.runtime[index]
        if self.get_in_deg:
            return cat_feats.long(), dense_feats.long(), src.long(), dst.long(), in_deg.long(), runtime.long()
        else:
            return cat_feats.long(), dense_feats.long(), src.long(), dst.long(), runtime.long()

    def read_runtime(self, runtimes):
        raise NotImplementedError()
    
class SymbolicChakraDataset(ChakraDataset):
    def read_runtime(self, runtimes):
        def _filename_to_dp(_filename):
            _filename = os.path.split(_filename)[-1]
            _filename = _filename.split('.')[0]
            terms = _filename.split("_")
            assert len(terms) == 13
            for i, term in enumerate(terms):
                terms[i] = int(term)
            dp = str(tuple(terms))
            return dp
        runtime_raws = runtimes
        # runtimes = dict()
        runtime = list()
        for i, runtime_raw in enumerate(runtime_raws):
            f = open(runtime_raw, 'r')
            runtime_raw = json.load(f)
            f.close()
            runtime_raws[i] = runtime_raw
        for file in self.chakra_files:
            dp = _filename_to_dp(file)
            runtime_this_file = list()
            for runtime_raw in runtime_raws:
                runtime_this_file.append(runtime_raw[dp])
            runtime.append(torch.tensor(runtime_this_file))
        return runtime

def load_chakra_s_s32_m256_asym(train_val_ratio=0.8):
    dataset_root = "./data_files/chakra_raw/comp/raw"
    runtimes = [
        "./data_files/chakra_raw/results_comp.json"
    ]
    fullset = SymbolicChakraDataset(dataset_root, runtimes, only_sym=True)
    # train_range = range(int(len(fullset)*train_val_ratio))
    # val_range = range(int(len(fullset)*train_val_ratio), len(fullset))
    # train_set = Subset(fullset, train_range)
    # val_set = Subset(fullset, val_range)
    return fullset, None, None

class ChakraNodeToDAGLMNodeOnlyNodeType(ChakraNodeToDAGLMNode):
    def get_dense_feat(self):
        # override dense feat with dummy value
        return [1.0]

class ChakraToDAGLMGraphOnlyNodeType(ChakraToDAGLMGraph):
    def __init__(self, chakra_graph):
        super().__init__(chakra_graph)
        self.node_converter = ChakraNodeToDAGLMNodeOnlyNodeType(None)

class ChakraDatasetOnlyNodeType(ChakraDataset):
    def __getitem__(self, index):
        if self.preload:
            cat_feats, dense_feats, src, dst, in_deg, runtime = self.preloaded_data[index]
            if self.get_in_deg:
                return cat_feats, dense_feats, src, dst, in_deg, runtime
            else:
                return cat_feats, dense_feats, src, dst, runtime
        chakra_file = self.chakra_files[index]
        cat_feats, dense_feats, src, dst, in_deg = ChakraToDAGLMGraphOnlyNodeType(chakra_file).process()
        runtime = self.runtime[index]
        if self.get_in_deg:
            return cat_feats.long(), dense_feats.float(), src.long(), dst.long(), in_deg.long(), runtime.float()
        else:
            return cat_feats.long(), dense_feats.float(), src.long(), dst.long(), runtime.float()
        
class SymbolicChakraDatasetOnlyNodeType(ChakraDatasetOnlyNodeType):
    def read_runtime(self, runtimes):
        def _filename_to_dp(_filename):
            _filename = os.path.split(_filename)[-1]
            _filename = _filename.split('.')[0]
            terms = _filename.split("_")
            assert len(terms) == 5
            for i, term in enumerate(terms):
                terms[i] = int(term)
            dp = str(tuple(terms))
            return dp
        runtime_raws = runtimes
        # runtimes = dict()
        runtime = list()
        for i, runtime_raw in enumerate(runtime_raws):
            f = open(runtime_raw, 'r')
            runtime_raw = json.load(f)
            f.close()
            runtime_raws[i] = runtime_raw
        for file in self.chakra_files:
            dp = _filename_to_dp(file)
            runtime_this_file = list()
            for runtime_raw in runtime_raws:
                runtime_this_file.append(runtime_raw[dp])
            runtime.append(torch.tensor(runtime_this_file))
        return runtime

def load_chakra_s_s32_m256_asym_only_node_type(train_val_ratio=0.8):
    dataset_root = "./data_files/chakra/raw/s_s32_m256_asym"
    runtimes = [
        "./data_files/chakra/raw/s_s32_m256_asym/s_s32_m256_asym_runtime_1.json",
        "./data_files/chakra/raw/s_s32_m256_asym/s_s32_m256_asym_runtime_2.json",
        "./data_files/chakra/raw/s_s32_m256_asym/s_s32_m256_asym_runtime_3.json",
    ]
    fullset = SymbolicChakraDatasetOnlyNodeType(dataset_root, runtimes, only_sym=True)
    # train_range = range(int(len(fullset)*train_val_ratio))
    # val_range = range(int(len(fullset)*train_val_ratio), len(fullset))
    # train_set = Subset(fullset, train_range)
    # val_set = Subset(fullset, val_range)
    train_set = fullset
    val_set = fullset
    return train_set, val_set, None


if __name__ == '__main__':
    path = './data_files/chakra/raw/s_s32_m64_sym/1_1_64_1_0.0.et'
    converter = ChakraToDAGLMGraphOnlyNodeType(path)
    converted = converter.process()
    hook = 0
    
    dataset_root = "./data_files/chakra/raw/s_s32_m256_asym"
    runtimes = [
        "./data_files/chakra/raw/s_s32_m256_asym/s_s32_m256_asym_runtime_1.json",
        "./data_files/chakra/raw/s_s32_m256_asym/s_s32_m256_asym_runtime_2.json",
        "./data_files/chakra/raw/s_s32_m256_asym/s_s32_m256_asym_runtime_3.json",
    ]
    dataset = SymbolicChakraDatasetOnlyNodeType(dataset_root, runtimes)
    for dp in dataset:
        for dpp in dp:
            print(dpp.shape, end=" ")
        print("\n")
        hook = 2
    hook = 3
            