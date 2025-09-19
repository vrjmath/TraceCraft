import json
import math

MAX_INPUTS = 3
MAX_OUTPUTS = 2

def prod_shape(shape):
    if not shape:
        return 0
    result = 1
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            result *= dim
        else:
            # Non-positive or invalid dim treated as 0 size
            return 0
    return result

def extract_device(value):
    # Look for device string in value list or string
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and ('cuda' in v or 'cpu' in v):
                return v
    elif isinstance(value, str) and ('cuda' in value or 'cpu' in value):
        return value
    return "unknown"

def extract_scalar(value, dtype):
    # For booleans or scalar ints/floats
    if dtype.lower() == "bool":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            return 1 if value.lower() == "true" else 0
    if dtype.lower() in ["int", "int64", "float", "double"]:
        if isinstance(value, (int, float)):
            return float(value)
    # else no scalar
    return None

def extract_node_features(node):
    features = {}
    # 1) Operation name
    features['op_name'] = node.get('name', 'unknown')

    # 2) Inputs
    inputs = node.get('inputs', {})
    in_values = inputs.get('values', [])
    in_shapes = inputs.get('shapes', [])
    in_types = inputs.get('types', [])

    inputs_features = []
    for i in range(MAX_INPUTS):
        if i < len(in_types):
            dtype = in_types[i]
        else:
            dtype = "unknown"
        if i < len(in_shapes):
            shape = in_shapes[i]
        else:
            shape = []
        if i < len(in_values):
            value = in_values[i]
        else:
            value = None

        shape_size = prod_shape(shape)
        device = extract_device(value) if value is not None else "unknown"
        scalar_val = extract_scalar(value, dtype)

        inputs_features.append({
            "dtype": dtype,
            "shape_size": shape_size,
            "device": device,
            "scalar": scalar_val
        })
    features['inputs'] = inputs_features

    # 3) Outputs
    outputs = node.get('outputs', {})
    out_values = outputs.get('values', [])
    out_shapes = outputs.get('shapes', [])
    out_types = outputs.get('types', [])

    outputs_features = []
    for i in range(MAX_OUTPUTS):
        if i < len(out_types):
            dtype = out_types[i]
        else:
            dtype = "unknown"
        if i < len(out_shapes):
            shape = out_shapes[i]
        else:
            shape = []
        if i < len(out_values):
            value = out_values[i]
        else:
            value = None

        shape_size = prod_shape(shape)
        device = extract_device(value) if value is not None else "unknown"

        outputs_features.append({
            "dtype": dtype,
            "shape_size": shape_size,
            "device": device
        })
    features['outputs'] = outputs_features

    # 4) Optional op_schema (string)
    attrs = node.get('attrs', [])
    op_schema = ""
    for attr in attrs:
        if attr.get('name') == 'op_schema':
            op_schema = attr.get('value', '')
            break
    features['op_schema'] = op_schema if op_schema else "unknown"

    return features

def process_json_file(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    for node in nodes:
        node['feature_vector'] = extract_node_features(node)

    with open(output_path, 'w') as f_out:
        json.dump(data, f_out, indent=2)

if __name__ == "__main__":
    input_json_file = "rank-0.json"   # your input filename here
    output_json_file = "output_features.json"  # your output filename here

    process_json_file(input_json_file, output_json_file)
    print(f"Processed nodes saved to {output_json_file}")
