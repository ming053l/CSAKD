# First, we simulate a 256x256x172 HSI cube since we do not have the actual .npy file.
import  numpy as np, glob
# simulated_hsi_data = np.random.rand(256, 256, 172)
dx = glob.glob('../../Fusion_data/train/*GT*')
# print(dx)
# raise
# Path to the .npy file (replace with the actual path of the .npy file)
npy_file_path = dx[0]

simulated_hsi_data = np.load(npy_file_path)

# Now we will use the spectral python (spy) package to display the cube.
# Importing necessary functions from spectral
from spectral import imshow, view_cube

# Using view_cube to visualize the data as a cube.
# Note that 'view_cube' requires an X server to be running and may not work in all environments.
# If it doesn't work here, you would need to run this code on your local machine.
view = view_cube(simulated_hsi_data, bands=[29, 14, 7])
# view.show()
