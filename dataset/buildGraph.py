import re
import json
import torch
from torch_geometric.data import Data
from collections import defaultdict

def normalize_name(name):
    # Remove trailing underscore + number, e.g. "all_gather_into_tensor_1" -> "all_gather_into_tensor"
    return re.sub(r'_\d+$', '', name)

def extract_attr(attrs, key):
    # attrs is a list of dicts {"name":..., "int64Val":..., ...}
    for attr in attrs:
        if attr.get("name") == key:
            # Return first non-null value for the keys possible in your JSON
            for val_key in ["int64Val", "uint64Val", "int32Val", "stringVal"]:
                if val_key in attr:
                    val = attr[val_key]
                    if isinstance(val, str) and val.isdigit():
                        return int(val)
                    elif isinstance(val, int):
                        return val
                    else:
                        # For stringVal or others, return as is (you can customize)
                        return val
            # If it's a boolVal
            if "boolVal" in attr:
                return int(attr["boolVal"])
    return None

def build_graph_from_jsonl(jsonl_path, output_graph_path, output_attr_map_path):
    node_name_to_idx = dict()
    nodes_data = []
    edges = []

    # Temporary storage for nodes by id (string)
    nodes_by_id = dict()

    with open(jsonl_path, "r") as f:
        for line in f:
            node = json.loads(line)
            node_id = node.get("id")
            name = node.get("name")
            norm_name = normalize_name(name)

            # Add to mapping if new
            if norm_name not in node_name_to_idx:
                node_name_to_idx[norm_name] = len(node_name_to_idx)

            nodes_by_id[node_id] = {
                "norm_name": norm_name,
                "type": node.get("type"),
                "attrs": node.get("attr", []),
                "dataDeps": node.get("dataDeps", []),
                "orig_name": name,
                "orig_id": node_id,
            }

    # Now build node attribute tensor and edges
    # We'll store node attributes in order of sorted node ids to maintain consistency
    sorted_node_ids = sorted(nodes_by_id.keys(), key=lambda x: int(x))

    node_attrs_list = []
    node_id_to_idx = dict()
    for idx, nid in enumerate(sorted_node_ids):
        node_id_to_idx[nid] = idx

    for nid in sorted_node_ids:
        node = nodes_by_id[nid]
        norm_name = node["norm_name"]
        ntype = node["type"]
        attrs = node["attrs"]

        # Map type to 0 or 1
        if ntype == "COMP_NODE":
            node_type_val = 0
            num_ops = 0#extract_attr(attrs, "num_ops")
            tensor_size = 0#extract_attr(attrs, "tensor_size")
            if num_ops is None or tensor_size is None:
                print(f"ERROR: COMP_NODE id {nid} missing num_ops or tensor_size")
                num_ops = -1 if num_ops is None else num_ops
                tensor_size = -1 if tensor_size is None else tensor_size
            node_attr = torch.tensor([
                node_type_val,
                node_name_to_idx[norm_name]
            ], dtype=torch.long)
        elif ntype == "COMM_COLL_NODE":
            node_type_val = 1
            comm_type = 0#extract_attr(attrs, "comm_type")
            comm_size = 0#extract_attr(attrs, "comm_size")
            if comm_type is None or comm_size is None:
                print(f"ERROR: COMM_COLL_NODE id {nid} missing comm_type or comm_size")
                comm_type = -1 if comm_type is None else comm_type
                comm_size = -1 if comm_size is None else comm_size
            node_attr = torch.tensor([
                node_type_val,
                node_name_to_idx[norm_name]
            ], dtype=torch.long)
        else:
            # Skip nodes that are not COMP_NODE or COMM_COLL_NODE
            print(f"WARNING: Node id {nid} has unsupported type {ntype}, skipping.")
            continue

        node_attrs_list.append(node_attr)

        # Add edges from dataDeps
        for parent_id in node.get("dataDeps", []):
            if parent_id not in node_id_to_idx:
                print(f"WARNING: dataDeps parent {parent_id} not found for node {nid}")
                continue
            edges.append([node_id_to_idx[parent_id], node_id_to_idx[nid]])

    if not node_attrs_list:
        raise RuntimeError("No nodes processed. Check input data.")

    node_attr_tensor = torch.stack(node_attrs_list)  # shape [num_nodes, 4]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, num_edges]

    data = Data(x=node_attr_tensor, edge_index=edge_index)

    torch.save(data, output_graph_path)
    print(f"Graph saved to {output_graph_path}")

    # Save node name to idx map to txt file
    with open(output_attr_map_path, "w") as f:
        for name, idx in sorted(node_name_to_idx.items(), key=lambda x: x[1]):
            f.write(f"{name}\t{idx}\n")

    print(f"Node name to index map saved to {output_attr_map_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python build_graph.py <input_jsonl> <output_graph_pth> <output_attr_map_txt>")
        sys.exit(1)

    build_graph_from_jsonl(sys.argv[1], sys.argv[2], sys.argv[3])
