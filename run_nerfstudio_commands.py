import subprocess

checkpoint_path = r"D:\Thesis\Stonehenge_new\nerfstudio_colmap_aligned\nerfstudio_output\trained_model\colmap_data\splatfacto\2024-12-01_175414\config.yml"
output_path = r"D:\Thesis\Stonehenge_new\nerfstudio_colmap_aligned\renders"

command = [
    "ns-export",
    "--load-dir", checkpoint_path, 
    "--output-dir", output_path, 
    "--format", "blender"
]

# Run the command
try:
    subprocess.run(command, check=True)
    print("Export completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred during export: {e}")
except FileNotFoundError:
    print("The 'ns-export' command was not found. Make sure Nerfstudio is installed and in your PATH.")