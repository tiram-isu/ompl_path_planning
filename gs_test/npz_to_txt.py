import numpy as np

# Load the .npz file
npz_data = np.load('/app/gs_test/test_garden.npz')

# Assuming the .npz file contains multiple arrays, you can access them by their keys
for key in npz_data.files:
    array_data = npz_data[key]
    
    # Save each array to a .txt file
    # Modify the file name as needed, e.g., "array_name.txt"
    np.savetxt(f'{key}.txt', array_data)
