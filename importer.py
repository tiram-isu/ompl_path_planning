import torch
import struct
import numpy as np
from typing import Dict, Optional, Tuple

def load_gaussians_from_nerfstudio_ckpt(ckpt_path: str, device: str = "cuda") -> Dict:
    """
    Load Gaussian parameters from a Nerfstudio checkpoint.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    gauss_params = checkpoint.get("pipeline", {})
    
    required_keys = [
        "_model.gauss_params.means",
        "_model.gauss_params.scales",
        "_model.gauss_params.quats",
        "_model.gauss_params.opacities",
        "_model.gauss_params.features_dc",
    ]
    
    gaussian_data = {
        key.split(".")[-1]: gauss_params[key].cpu().numpy()
        for key in required_keys if key in gauss_params
    }
    
    missing_keys = set(required_keys) - set(gaussian_data.keys())
    if missing_keys:
        raise KeyError(f"Missing keys in pipeline: {', '.join(missing_keys)}")
    
    return gaussian_data

def load_gaussians_from_ply(input_ply_path: str) -> Optional[Dict]:
    """
    Extract Gaussian parameters from a binary or ASCII .ply file.
    """
    try:
        with open(input_ply_path, 'rb') as ply_file:
            content = ply_file.read()
        
        header_end = content.find(b'end_header\n')
        if header_end == -1:
            raise ValueError("Invalid .ply file: No end_header found.")
        
        header = content[:header_end + len(b'end_header\n')].decode('utf-8')
        body = content[header_end + len(b'end_header\n'):]
        is_little_endian = "binary_little_endian" in header
        endian_char = '<' if is_little_endian else '>'

        vertex_properties, vertex_count = __parse_ply_header(header)
        return __parse_ply_body(vertex_properties, vertex_count, body, endian_char)
    
    except Exception as e:
        print(f"Error loading .ply file: {e}")
        return None

def __parse_ply_header(header: str) -> Tuple[list, int]:
    """
    Parse the header of a .ply file to extract vertex properties and count.
    """
    vertex_properties = []
    vertex_count = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
        elif line.startswith("property"):
            vertex_properties.append(line.split()[2])
    return vertex_properties, vertex_count

def __parse_ply_body(vertex_properties: list, vertex_count: int, body: bytes, endian_char: str) -> Optional[Dict]:
    """
    Parse the binary section of a .ply file and structure Gaussian data.
    """
    data = {prop: [] for prop in vertex_properties}
    offset = 0
    for _ in range(vertex_count):
        for prop in vertex_properties:
            fmt, size = __property_format_and_size(prop, endian_char)
            if fmt is None:
                offset += size  # Skip unsupported properties
                continue
            value = struct.unpack(fmt, body[offset:offset + size])[0]
            offset += size
            data[prop].append(value)
    
    return __structure_gaussian_data(data)

def __property_format_and_size(property_name: str, endian_char: str) -> Optional[Tuple[str, int]]:
    """
    Get struct format and size for a given .ply property name.
    """
    if property_name.startswith(("f_dc_", "scale_", "rot_", "opacity", "x", "y", "z", "nx", "ny", "nz")):
        return f"{endian_char}f", 4
    return None, 4  # Skip unsupported properties

def __structure_gaussian_data(data: Dict) -> Optional[Dict]:
    """
    Structure raw .ply data into a dictionary of Gaussian parameters.
    """
    try:
        return {
            "means": np.column_stack((data["x"], data["y"], data["z"])),
            "scales": np.column_stack((data["scale_0"], data["scale_1"], data["scale_2"])),
            "quats": np.column_stack((data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"])),
            "opacities": np.array(data["opacity"]),
            "features_dc": np.column_stack([data[f"f_dc_{i}"] for i in range(3)]),
        }
    except KeyError as e:
        print(f"Error structuring Gaussian data: {e}")
        return None
    
# Convert to .txt (for debugging only, not used in the main script)

def convert_checkpoint_to_txt(checkpoint_path: str, output_txt_path: str) -> None:
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

def convert_ply_to_readable_txt(input_ply_path: str, output_txt_path: str) -> None:
    """
    Convert a .ply file (binary or ASCII) into a readable text format.
    """
    try:
        with open(input_ply_path, 'rb') as ply_file:
            content = ply_file.read()
        
        header_end = content.find(b'end_header\n')
        if header_end == -1:
            raise ValueError("Invalid .ply file: No end_header found.")
        
        header = content[:header_end + len(b'end_header\n')].decode('utf-8')
        body = content[header_end + len(b'end_header\n'):]
        readable_body = __parse_binary_ply(header, body) if "binary" in header else body.decode('utf-8')
        
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write("# Converted .ply file to .txt\n")
            txt_file.write(header)
            txt_file.write(readable_body)
        
        print(f"Successfully converted '{input_ply_path}' to '{output_txt_path}'.")
    except Exception as e:
        print(f"Error converting .ply to text: {e}")

def __parse_binary_ply(header: str, body: bytes) -> str:
    """
    Parse binary .ply file and return a readable string representation.
    """
    vertex_properties, vertex_count = __parse_ply_header(header)
    endian_char = '<' if "binary_little_endian" in header else '>'
    
    readable_data = []
    offset = 0
    for _ in range(vertex_count):
        instance_data = []
        for prop in vertex_properties:
            fmt, size = __property_format_and_size(prop, endian_char)
            if fmt is None:
                offset += size
                continue
            instance_data.append(str(struct.unpack(fmt, body[offset:offset + size])[0]))
            offset += size
        readable_data.append(" ".join(instance_data))
    
    return "\n".join(readable_data)
