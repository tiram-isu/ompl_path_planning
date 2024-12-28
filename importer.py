import torch
import struct
import numpy as np

def load_gaussians_from_nerfstudio_ckpt(ckpt_path, device="cuda"):
    """
    Load Gaussian parameters from a Nerfstudio Gsplat checkpoint.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    gauss_params = checkpoint.get("pipeline", {})
    required_keys = [
        "_model.gauss_params.means",
        "_model.gauss_params.scales",
        "_model.gauss_params.quats",
        "_model.gauss_params.opacities",
        "_model.gauss_params.features_dc",
        "_model.gauss_params.features_rest"
    ]
    gaussian_data = {
        key.split(".")[-1]: gauss_params[key].cpu().numpy()
        for key in required_keys
        if key in gauss_params
    }
    if len(gaussian_data) != len(required_keys):
        missing_keys = set(required_keys) - set(gaussian_data.keys())
        raise KeyError(f"Missing keys in pipeline: {', '.join(missing_keys)}")
    return gaussian_data

def load_gaussians_from_ply(input_ply_file):
    """
    Extract Gaussian parameters from a binary or ASCII .ply file.
    """
    try:
        with open(input_ply_file, 'rb') as ply_file:
            content = ply_file.read()

        header_end = content.find(b'end_header\n')
        if header_end == -1:
            raise ValueError("Invalid .ply file: No end_header found.")

        header = content[:header_end + len(b'end_header\n')].decode('utf-8')
        body = content[header_end + len(b'end_header\n'):]
        is_little_endian = "binary_little_endian" in header
        endian_char = '<' if is_little_endian else '>'

        vertex_properties, vertex_count = parse_ply_header(header)

        data = {prop: [] for prop in vertex_properties}
        offset = 0
        for _ in range(vertex_count):
            for prop in vertex_properties:
                fmt, size = property_format_and_size(prop, endian_char)
                value = struct.unpack(fmt, body[offset:offset + size])[0]
                offset += size
                data[prop].append(value)

        for key in data:
            data[key] = np.array(data[key])

        return structure_gaussian_data(data)

    except Exception as e:
        print(f"Error loading .ply file: {e}")
        return None

def parse_ply_header(header):
    """
    Parse the header of a .ply file to extract vertex properties and counts.
    """
    lines = header.splitlines()
    vertex_properties = []
    vertex_count = 0
    for line in lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
        elif line.startswith("property"):
            vertex_properties.append(line.split()[2])
    return vertex_properties, vertex_count

def property_format_and_size(property_name, endian_char):
    """
    Get the struct format and size for a given .ply property name.
    """
    if property_name.startswith(("f_dc_", "f_rest_", "scale_", "rot_", "opacity", "x", "y", "z", "nx", "ny", "nz")):
        return f"{endian_char}f", 4  # All are floats
    raise ValueError(f"Unsupported property: {property_name}")

def structure_gaussian_data(data):
    """
    Structure raw .ply data into a dictionary of Gaussian parameters.
    """
    return {
        "means": np.column_stack((data["x"], data["y"], data["z"])),
        "scales": np.column_stack((data["scale_0"], data["scale_1"], data["scale_2"])),
        "quats": np.column_stack((data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"])),
        "opacities": np.array(data["opacity"]),
        "features_dc": np.column_stack([data[f"f_dc_{i}"] for i in range(3)]),
        "features_rest": np.column_stack([data[f"f_rest_{i}"] for i in range(40)])  # Assumes f_rest_0 to f_rest_39
    }

def convert_checkpoint_to_txt(checkpoint_path, output_txt_path):
    """
    Convert a PyTorch checkpoint to a human-readable text file.
    """
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))

    with open(output_txt_path, 'w') as f:
        f.write("Checkpoint details:\n")
        for key in checkpoint.keys():
            f.write(f"  - {key}\n")
        f.write("\nModel Parameters:\n")
        for name, param in model.items():
            f.write(f"\n{name}:\n  Shape: {param.shape if isinstance(param, torch.Tensor) else 'N/A'}\n")
            if isinstance(param, torch.Tensor):
                f.write(f"  Values:\n{param.cpu().numpy()}\n")
    print(f"Checkpoint written to {output_txt_path}")

def convert_ply_to_readable_txt(input_ply_file, output_txt_file):
    """
    Convert a .ply file (binary or ASCII) into a readable text format.
    """
    try:
        with open(input_ply_file, 'rb') as ply_file:
            content = ply_file.read()

        header_end = content.find(b'end_header\n')
        if header_end == -1:
            raise ValueError("Invalid .ply file: No end_header found.")

        header = content[:header_end + len(b'end_header\n')].decode('utf-8')
        body = content[header_end + len(b'end_header\n'):]
        readable_body = parse_binary_ply(header, body) if "binary" in header else body.decode('utf-8')

        with open(output_txt_file, 'w') as txt_file:
            txt_file.write("# Converted .ply file to .txt\n")
            txt_file.write(header)
            txt_file.write(readable_body)

        print(f"Successfully converted '{input_ply_file}' to '{output_txt_file}'.")

    except Exception as e:
        print(f"Error converting .ply to text: {e}")

def parse_binary_ply(header, body):
    """
    Parse the binary section of a .ply file.
    """
    vertex_properties, vertex_count = parse_ply_header(header)
    is_little_endian = "binary_little_endian" in header
    endian_char = '<' if is_little_endian else '>'

    readable_data = ""
    offset = 0
    for _ in range(vertex_count):
        instance_data = []
        for prop in vertex_properties:
            fmt, size = property_format_and_size(prop, endian_char)
            value = struct.unpack(fmt, body[offset:offset + size])[0]
            offset += size
            instance_data.append(value)
        readable_data += " ".join(map(str, instance_data)) + "\n"
    return readable_data
