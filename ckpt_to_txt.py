import torch

# Define the function to load the checkpoint and convert to human-readable txt
def convert_checkpoint_to_txt(checkpoint_path, output_txt_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Open the output txt file
    with open(output_txt_path, 'w') as f:
        # Writing general information about the checkpoint
        f.write("Checkpoint details:\n")
        f.write(f"Checkpoint contains the following keys:\n")
        for key in checkpoint.keys():
            f.write(f"  - {key}\n")

        # Extract and write model parameters (weights, biases, etc.)
        f.write("\nModel Parameters (Weights & Biases):\n")
        
        # Assuming 'model' contains the actual model parameters
        # If the checkpoint is saved under a different key (e.g., 'model_state_dict')
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            model = checkpoint['model_state_dict']
        else:
            model = checkpoint

        # Iterate through the model parameters and write them to the file
        for name, param in model.items():
            f.write(f"\n{name}:\n")
            
            # Check if param is a tensor
            if isinstance(param, torch.Tensor):
                f.write(f"  Shape: {param.shape}\n")
                # Writing the actual tensor values as a numpy array
                f.write(f"  Values:\n{param.cpu().numpy()}\n")
            else:
                # If param is not a tensor, write its value directly
                f.write(f"  Value: {param}\n")

        f.write("\n--- End of Model Parameters ---\n")
    
    print(f"Checkpoint data has been written to {output_txt_path}")

# Example usage:
checkpoint_path = '/app/gs_test/test_garden.ckpt'  # Replace with your actual checkpoint path
output_txt_path = '/app/gs_test/test_garden.txt'  # Desired output txt file

convert_checkpoint_to_txt(checkpoint_path, output_txt_path)
