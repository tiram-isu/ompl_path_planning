import numpy as np
import torch
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
from scipy.spatial.transform import Rotation as R
from OpenGL.arrays import vbo

def load_gaussians_from_nerfstudio_ckpt(ckpt_path, device="cuda"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    gauss_params = checkpoint["pipeline"]
    
    required_keys = [
        "_model.gauss_params.means", 
        "_model.gauss_params.scales", 
        "_model.gauss_params.quats", 
        "_model.gauss_params.opacities",
        "_model.gauss_params.features_dc", 
        "_model.gauss_params.features_rest"
    ]
    
    gaussian_data = {}
    for key in required_keys:
        if key not in gauss_params:
            raise KeyError(f"Expected key '{key}' in 'pipeline' but found none.")
        gaussian_data[key.split(".")[-1]] = gauss_params[key].to(device)

    return gaussian_data

def normalize_colors(features_dc):
    min_val = features_dc.min()
    max_val = features_dc.max()
    normalized_colors = (features_dc - min_val) / (max_val - min_val)
    
    # Apply clamp operation while keeping it as a tensor
    normalized_colors = torch.clamp(normalized_colors, 0, 1)

    # Convert the result to NumPy array after clamping
    return normalized_colors.cpu().numpy()

def render_gaussians_on_gpu(gaussian_data):
    # Extract the necessary data from Gaussian parameters
    means = gaussian_data["means"].cpu().numpy()
    scales = gaussian_data["scales"].cpu().numpy()
    quats = gaussian_data["quats"].cpu().numpy()
    features_dc = gaussian_data["features_dc"].cpu().numpy()
    
    # Normalize the colors using PyTorch for the computation
    normalized_colors = normalize_colors(features_dc)

    # Create a window using GLUT
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
    glut.glutCreateWindow(b"GPU Rendering with PyOpenGL")
    
    # Set up OpenGL context
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    
    # Set up perspective
    glu.gluPerspective(45, 1, 0.1, 50.0)
    gl.glTranslatef(0.0, 0.0, -10)

    def draw_scene():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        
        for i in range(len(means)):
            # Set up transformations for each ellipsoid (based on Gaussian params)
            gl.glPushMatrix()

            # Translate to the mean position
            gl.glTranslatef(*means[i])

            # Apply rotation using quaternion
            rotation_matrix = R.from_quat(quats[i]).as_matrix()
            gl.glMultMatrixf(rotation_matrix.flatten())

            # Apply scaling
            gl.glScalef(scales[i][0], scales[i][1], scales[i][2])

            # Set color based on normalized features
            color = normalized_colors[i]
            gl.glColor3f(color[0], color[1], color[2])

            # Draw a unit sphere to represent the ellipsoid
            glut.glutSolidSphere(1.0, 20, 20)

            gl.glPopMatrix()

        glut.glutSwapBuffers()

    # Set the display function to render the scene
    glut.glutDisplayFunc(draw_scene)

    # Enter the GLUT main loop to render the scene
    glut.glutMainLoop()

# Example usage
ckpt_path = "/app/models/stonehenge.ckpt"
device = "cuda"  # "cuda" for GPU, "cpu" for CPU
gaussian_data = load_gaussians_from_nerfstudio_ckpt(ckpt_path, device=device)

# Start rendering on GPU using OpenGL
render_gaussians_on_gpu(gaussian_data)
