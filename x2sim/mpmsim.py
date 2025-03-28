#!/usr/bin/env python3
"""
MPM simulation module for x2sim pipeline
"""
import os
import sys
import json
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("x2sim/mpm_sim.log"), logging.StreamHandler()]
)
logger = logging.getLogger("mpm_sim")

# Workaround to access Taichi MPM - Updated path
current_dir = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
taichi_mpm_path = os.path.join(current_dir, 'taichi_mpm')

if os.path.exists(taichi_mpm_path):
    sys.path.append(taichi_mpm_path)
    logger.info(f"Added Taichi MPM path: {taichi_mpm_path}")
else:
    logger.warning(f"Taichi MPM path not found: {taichi_mpm_path}")
    logger.warning("Using current working directory for imports")

try:
    # Try to import run_mpm from the taichi_mpm directory
    from taichi_mpm.run_mpm import run_collision
    logger.info("Successfully imported run_collision from taichi_mpm.run_mpm")
except ImportError as e:
    logger.warning(f"Failed to import from taichi_mpm.run_mpm: {e}")
    logger.warning("Trying direct import...")
    try:
        # Fallback to direct import
        import run_mpm
        logger.info("Successfully imported run_mpm directly")
    except ImportError as e:
        logger.error(f"Failed to import run_mpm: {e}")
        logger.error("Make sure the Taichi MPM module is available")

def preprocess_point_cloud(input_path, output_path=None, max_points=100000,
                       domain_size=[[0.1, 1.9], [0.1, 1.9], [0.1, 1.9]], 
                       target_occupancy=0.4, point_e_rotation=False):
    """
    Preprocess point cloud by:
    1. Downsampling to reduce the number of points
    2. Removing statistical outliers
    3. Rotating to the desired orientation
    4. Scaling to target occupancy of domain
    5. Centering and translating to fit within domain
    
    Args:
        input_path (str): Path to the input point cloud file (.ply)
        output_path (str, optional): Path to save the processed point cloud file.
                                    If None, will create a new path overwriting input.
        max_points (int, optional): Maximum number of points in the output. Default is 100000.
        domain_size (list): Domain boundaries as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        target_occupancy (float): Target size as fraction of domain size (default: 0.4)
        point_e_rotation (bool): Whether to use Point-E specific rotation (default: False)
                                   
    Returns:
        str: Path to the processed point cloud file
    """
    
    logger = logging.getLogger("mpm_sim")
    
    # Set output path if not provided (overwrite input)
    if output_path is None:
        output_path = input_path
            
    logger.info(f"Preprocessing point cloud: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Target max points: {max_points}")
    logger.info(f"Domain size: {domain_size}")
    logger.info(f"Target occupancy: {target_occupancy}")
    
    try:
        import open3d as o3d
        
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(input_path)
        original_points = len(np.asarray(pcd.points))
        logger.info(f"Original point cloud has {original_points} points")
        
        # First step: Downsample if necessary
        if original_points > max_points:
            # Calculate voxel size for downsampling
            # Start with a reasonable size and iteratively increase if needed
            voxel_size = 0.01
            while True:
                downsampled_pcd = pcd.voxel_down_sample(voxel_size)
                current_points = len(np.asarray(downsampled_pcd.points))
                
                logger.info(f"Downsampled to {current_points} points with voxel size {voxel_size}")
                
                if current_points <= max_points or voxel_size > 1.0:
                    pcd = downsampled_pcd
                    break
                    
                voxel_size *= 1.5
        
        # Second step: Remove outliers
        logger.info("Removing statistical outliers...")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        pcd = pcd.select_by_index(ind)
        cleaned_points = len(np.asarray(pcd.points))
        logger.info(f"After outlier removal: {cleaned_points} points")
        
        # Calculate initial properties
        bbox = pcd.get_axis_aligned_bounding_box()
        current_dimensions = bbox.get_extent()
        centroid = bbox.get_center()
        logger.info(f"Current bounding box: {bbox}")
        logger.info(f"Current dimensions: {current_dimensions}")
        logger.info(f"Current centroid: {centroid}")
        
        # Third step: Rotate the point cloud
        if point_e_rotation:
            # Rotate the point cloud 90 degrees about the y-axis, and 180 about x-axis (Point-E)
            logger.info("Applying Point-E rotation (90° about y-axis, 180° about x-axis)")
            R = pcd.get_rotation_matrix_from_xyz((np.pi, np.pi / 2, 0))
        else:
            # Custom rotation (180° about z-axis)
            logger.info("Applying custom rotation (180° about z-axis)")
            R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
            
        pcd.rotate(R, center=centroid)
        
        # Fourth step: Scale to target size
        # Calculate target dimensions (based on occupancy percentage of domain size)
        domain_extent = np.array([d[1] - d[0] for d in domain_size])
        target_dimensions = domain_extent * target_occupancy
        
        # Determine scaling factor
        scaling_factors = target_dimensions / current_dimensions
        scale_factor = min(scaling_factors)  # Use the smallest scaling factor to fit all dimensions
        
        logger.info(f"Domain extent: {domain_extent}")
        logger.info(f"Target dimensions: {target_dimensions}")
        logger.info(f"Scaling factors: {scaling_factors}")
        logger.info(f"Final scale factor: {scale_factor}")
        
        # Scale the point cloud
        pcd.scale(scale_factor, center=centroid)
        
        # Fifth step: Move the point cloud to the ground (lowest y value to domain floor)
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        translation = [0, domain_size[1][0] - min_bound[1], 0]
        logger.info(f"Ground translation: {translation}")
        pcd.translate(translation)
        
        # Recalculate properties after rotation, scaling, and translation
        bbox = pcd.get_axis_aligned_bounding_box()
        current_dimensions = bbox.get_extent()
        centroid = bbox.get_center()
        logger.info(f"Bounding box after rotation, scaling, and ground placement: {bbox}")
        logger.info(f"Dimensions after rotation, scaling, and ground placement: {current_dimensions}")
        logger.info(f"Centroid after rotation, scaling, and ground placement: {centroid}")
        
        # Sixth step: Center the point cloud in the domain horizontally (x and z)
        domain_center = np.array([
            (domain_size[0][0] + domain_size[0][1]) / 2,  # x center
            0,  # Keep y unchanged
            (domain_size[2][0] + domain_size[2][1]) / 2   # z center
        ])
        
        horizontal_translation = [
            domain_center[0] - centroid[0],
            0,
            domain_center[2] - centroid[2]
        ]
        
        logger.info(f"Horizontal centering translation: {horizontal_translation}")
        pcd.translate(horizontal_translation)
        
        # Final properties
        final_bbox = pcd.get_axis_aligned_bounding_box()
        final_dimensions = final_bbox.get_extent()
        final_centroid = final_bbox.get_center()
        
        logger.info(f"Final bounding box: {final_bbox}")
        logger.info(f"Final dimensions: {final_dimensions}")
        logger.info(f"Final centroid: {final_centroid}")
        
        # Save the preprocessed point cloud
        logger.info(f"Saving preprocessed point cloud to {output_path}")
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
        
        return output_path
        
    except ImportError as e:
        logger.error(f"Failed to import Open3D: {e}")
        logger.error("This function requires Open3D for full preprocessing.")
        logger.error("Please install Open3D with: pip install open3d")
        return input_path
    except Exception as e:
        logger.error(f"Error in point cloud preprocessing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Returning original point cloud path without preprocessing")
        return input_path


def create_empty_ply(output_path):
    """
    Create an empty PLY file with minimal structure.
    
    Args:
        output_path (str): Path to save the empty PLY file
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create minimal PLY header
    ply_content = """ply
        format ascii 1.0
        element vertex 0
        property float x
        property float y
        property float z
        element face 0
        property list uchar int vertex_indices
        end_header
        """
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(ply_content)
        
def gen_sim(input_data: str):
    """Generate MPM simulation using the provided input data"""
    logger.info("Starting MPM simulation...")
    
    # Create output directory if it doesn't exist
    save_path = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(save_path, exist_ok=True)
    
    # Basic simulation parameters
    id_range = [0, 1]  # Number of simulations to do
    domain_size = 2.0  # Size of the domain
    sim_space = [[0.1, 1.9], [0.1, 1.9], [0.1, 1.9]]  # Simulation boundaries
    sim_resolution = [32, 32, 32]  # Simulation resolution
    nsteps = 500  # Number of timesteps
    mpm_dt = 0.0025  # Time intervals
    gravity = [0, -9.81, 0]  # Gravity
    wall_friction = 0.5  # Wall friction
    is_realtime_vis = False
    is_save_animation = True  # Save animations
    skip = 1  # Frame capture frequency
    generate = True  # Generate from point cloud
    
    # Check for fluid simulation parameters
    fluid_path = os.path.join(os.getcwd(), 'x2sim/fluid_simulation.json')
    logger.info(f"Looking for fluid simulation parameters at: {fluid_path}")
    
    if os.path.exists(fluid_path):
        try:
            with open(fluid_path, "r") as json_file:
                data = json.load(json_file)
            
            logger.info("Fluid simulation parameters loaded successfully")
            
            # Extract parameters from the JSON
            if "cubes" in data and isinstance(data["cubes"], list) and len(data["cubes"]) > 0:
                cubes = data["cubes"]
                logger.info(f"Using cubes configuration: {cubes}")
            else:
                logger.warning("Invalid or missing 'cubes' in fluid simulation data")
                cubes = [[0.11, 0.2, 0.12, 0.4, 0.6, 0.5], [1.5, 0.2, 1.5, 0.3, 0.6, 0.3]]
            
            if "velocity" in data and isinstance(data["velocity"], list) and len(data["velocity"]) > 0:
                velocity_for_cubes = data["velocity"]
                logger.info(f"Using velocity configuration: {velocity_for_cubes}")
            else:
                logger.warning("Invalid or missing 'velocity' in fluid simulation data")
                velocity_for_cubes = [[1.5, 1.0, 0.5], [-1.5, 1.0, 0.5]]
        except Exception as e:
            logger.error(f"Error loading fluid simulation parameters: {e}")
            cubes = [[0.11, 0.2, 0.12, 0.4, 0.6, 0.5], [1.5, 0.2, 1.5, 0.3, 0.6, 0.3]]
            velocity_for_cubes = [[1.5, 1.0, 0.5], [-1.5, 1.0, 0.5]]
    else:
        logger.warning("Fluid simulation parameters not found, using defaults")
        cubes = [[0.11, 0.2, 0.12, 0.4, 0.6, 0.5], [1.5, 0.2, 1.5, 0.3, 0.6, 0.3]]
        velocity_for_cubes = [[1.5, 1.0, 0.5], [-1.5, 1.0, 0.5]]
    
    # Check for point cloud file
    pc_path = os.path.join(os.getcwd(), 'outputs/preprocessed.ply')
    logger.info(f"Using point cloud from: {pc_path}")
    if not os.path.exists(pc_path):
        logger.error(f"Point cloud file not found at: {pc_path}")
        return f"Error: Point cloud file not found at {pc_path}"
    
    # Preprocess and downsample the point cloud
    preprocessed_pc_path = preprocess_point_cloud(pc_path, max_points=100000)
    logger.info(f"Using preprocessed point cloud from: {preprocessed_pc_path}")
    
    # Set obstacle parameters
    elastic = False
    velocity_for_obstacle = [0.0, 0.0, 0.0]
    
    # Create the inputs.json file for Taichi MPM
    inputs_dict = {
        "save_path": save_path,
        "id_range": id_range,
        "wall_friction": wall_friction,
        "domain_size": domain_size,
        "sim_space": sim_space,
        "sim_resolution": sim_resolution,
        "nsteps": nsteps,
        "mpm_dt": mpm_dt,
        "gravity": gravity,
        "gen_from_pc": {
            "generate": generate,
            "sim_inputs": [
                {
                    "id": id_range[0],
                    "mass": {
                        "cubes": cubes,
                        "velocity_for_cubes": velocity_for_cubes
                    },
                    "obstacles": [
                        {
                            "pc_path": pc_path,
                            "elastic": elastic,
                            "velocity_for_obstacle": velocity_for_obstacle
                        }
                    ]
                }
            ]
        },
        "visualization": {
            "is_realtime_vis": is_realtime_vis,
            "is_save_animation": is_save_animation,
            "skip": skip
        }
    }

    # Write the inputs to a JSON file
    inputs_filename = "x2sim/inputs.json"
    try:
        with open(inputs_filename, 'w') as json_file:
            json.dump(inputs_dict, json_file, indent=2)
        
        inputs_path = os.path.join(os.getcwd(), inputs_filename)
        if os.path.exists(inputs_path):
            logger.info(f"Inputs file created successfully: {inputs_path}")
        else:
            logger.error(f"Failed to create inputs file at: {inputs_path}")
            return f"Error: Failed to create inputs file at {inputs_path}"
    except Exception as e:
        logger.error(f"Error creating inputs file: {e}")
        return f"Error creating inputs file: {str(e)}"
    
    # Run the Taichi MPM simulation
    try:
        logger.info("Starting Taichi collision simulation")
        
        # Load the inputs.json file
        with open(inputs_path) as f:
            inputs = json.load(f)
        
        # Run the simulation for each ID in the range
        for i in range(id_range[0], id_range[1]):
            logger.info(f"Running collision simulation for ID {i}")
            
            # Use the correct run_collision function based on which import method succeeded
            if 'run_collision' in globals():
                data = run_collision(i, inputs, follow_taichi_coord=True)
            elif 'run_mpm' in globals() and hasattr(run_mpm, 'run_collision'):
                data = run_mpm.run_collision(i, inputs, follow_taichi_coord=True)
            else:
                raise ImportError("Neither run_collision nor run_mpm.run_collision is available")
                
            logger.info(f"Collision simulation for ID {i} completed")
        
        # Check if simulation output was created
        expected_output = os.path.join(save_path, f"collision_{id_range[0]}")
        if os.path.exists(expected_output) or os.path.exists(save_path):
            logger.info(f"Simulation outputs created at: {save_path}")
            return f"Successful simulation! Outputs saved to {save_path}"
        else:
            logger.warning(f"Simulation ran but outputs not found at expected location: {expected_output}")
            return f"Simulation completed but outputs may not have been created as expected"
        
    except Exception as e:
        logger.error(f"Error running Taichi MPM simulation: {e}")
        return f"Error running MPM simulation: {str(e)}"

if __name__ == "__main__":
    result = gen_sim("Test MPM simulation")
    print(result)