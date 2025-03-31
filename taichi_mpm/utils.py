import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import random
import pandas as pd
from typing import Union
import open3d as o3d
import os
import engine.mpm_solver


def generate_random_cube(
        space_size,
        cube_size_range,
):
    """
    Make a cube which is defined as,
    [x_start, y_start, z_start, z_len, y_len, z_len]

    space_size: a domain where cube can be generated e.g., ((0.2, 0.8), (0.2, 0.8))
    cube_size_range: a range that defines random cube size.
      It can be
        1) Size ranges are defined for all dims. (e.g., [[0.15, 0.3], [0.15, 0.3], [0.15, 0.3]]
        2) Or, if you want it to be squared shape, [0.3, 0.5] which is the range of squared-shaped cube
        that will be generated.

    """
    ndim = len(space_size)
    try:
        # if size ranges are defined for all dims
        cube_sizes = [random.uniform(min_max[0], min_max[1]) for min_max in cube_size_range]
    except:
        # if only one size range is defined (e.g., squared shape)
        size = random.uniform(cube_size_range[0], cube_size_range[1])
        cube_sizes = [size for _ in range(ndim)]
    cube_starts = [random.uniform(space_size[i][0], space_size[i][1] - cube_sizes[i]) for i in range(ndim)]
    return (*cube_starts, *cube_sizes)

def check_overlap(cube1, cube2, min_distance_between_cubes=0.0):
    ndim = int(len(cube1) / 2)
    for i in range(ndim):
        if cube1[i] - min_distance_between_cubes >= cube2[i] + cube2[i + ndim] or \
           cube1[i] + cube1[i + ndim] + min_distance_between_cubes <= cube2[i]:
            return False
    return True

def calculate_particles(cubes, density):
    ndim = len(cubes[0]) / 2
    if ndim == 3:
        total_volume = sum(c[3] * c[4] * c[5] for c in cubes)
    elif ndim == 2:
        total_volume = sum(c[2] * c[3] for c in cubes)
    else:
        raise ValueError("Only 2D and 3D dimensions are supported.")
    return total_volume * density

def generate_cubes(n,
                   space_size,
                   cube_size_range,
                   min_distance_between_cubes,
                   density,
                   max_particles=float('inf')):
    """
    Make none-overlapping n number of cubes which is defined as,
    [x_start, y_start, z_start, z_len, y_len, z_len]

    space_size: a domain where cube can be generated e.g., ((0.2, 0.8), (0.2, 0.8))
    cube_size_range: a range that defines random cube size.
      It can be
        1) Size ranges are defined for all dims. (e.g., [[0.15, 0.3], [0.15, 0.3], [0.15, 0.3]]
        2) Or, if you want it to be squared shape, [0.3, 0.5] which is the range of squared-shaped cube
        that will be generated.
    min_distance_between_cubes: separation distance between cubes
    density: n particle per volume (n-particles/m^3)
    max_particles: restrict the numer of particles that will be generated.
    """
    cubes = []
    attempts = 0
    resets = 0
    max_resets = 10000000  # You can adjust this limit as needed.

    while len(cubes) < n:
        new_cube = generate_random_cube(space_size, cube_size_range)

        # Check for overlap with existing cubes
        if any(check_overlap(new_cube, cube, min_distance_between_cubes=min_distance_between_cubes) for cube in cubes):
            # If there is an overlap, clear the cubes and count a reset.
            cubes.clear()
            resets += 1
            if resets > max_resets:
                raise Exception(f"Too many resets ({resets}), unable to place non-overlapping cubes.")
            # Continue to the next iteration, starting the process over.
            continue

        cubes.append(new_cube)
        if calculate_particles(cubes, density) > max_particles:
            return cubes[:-1]

        attempts += 1
        if attempts > 10000000:
            raise Exception(f"Cannot find non-overlapping cubes in {attempts} attempts")

    return cubes


def T(a):
    phi, theta = np.radians(32), np.radians(10)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5

def animation_from_npz(
        path,
        npz_name,
        save_name,
        boundaries,
        timestep_stride=5,
        colorful=True,
        follow_taichi_coord=True):
    """
    Create animation from NPZ file containing simulation trajectory.
    
    Args:
        path (str): Path to directory containing NPZ file
        npz_name (str): Name of NPZ file (without .npz extension)
        save_name (str): Name to use for saved animation
        boundaries (list): Simulation boundaries [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        timestep_stride (int): How many timesteps to skip between frames
        colorful (bool): Whether to color particles by velocity
        follow_taichi_coord (bool): Whether to follow Taichi coordinate convention
    """
    print(f"Loading NPZ file: {path}/{npz_name}.npz")
    data = np.load(f"{path}/{npz_name}.npz", allow_pickle=True)
    
    # Get positions data from the correct key (trajectory0)
    if npz_name in data:
        # Access the actual data from the NPZ
        trajectory_data = data[npz_name]
        
        # Check if trajectory_data is the expected shape (2,)
        if isinstance(trajectory_data, np.ndarray) and len(trajectory_data) >= 1:
            positions = trajectory_data[0]  # First element of tuple is positions
            print(f"Positions shape: {positions.shape}")
        else:
            raise ValueError(f"Unexpected data format in NPZ: {type(trajectory_data)}")
    else:
        # Default to first key if specific key not found
        print(f"Key '{npz_name}' not found. Available keys: {list(data.keys())}")
        key = list(data.keys())[0]
        if key in ['allow_pickle', 'pickle_kwargs']:
            key = list(data.keys())[1]  # Skip metadata keys
        
        trajectory_data = data[key]
        if isinstance(trajectory_data, np.ndarray) and len(trajectory_data) >= 1:
            positions = trajectory_data[0]
            print(f"Using positions from key '{key}'. Shape: {positions.shape}")
        else:
            raise ValueError(f"Could not find valid positions data in NPZ file")
    
    ndim = positions.shape[-1]
    print(f"Animation dimensions: {ndim}D")
    
    # Compute velocity magnitude for color bar
    if colorful:
        print("Computing velocity for colormap...")
        initial_vel = np.zeros(positions[0].shape)
        initial_vel = initial_vel.reshape((1, initial_vel.shape[0], initial_vel.shape[1]))
        vel = positions[1:] - positions[:-1]
        vel = np.concatenate((initial_vel, vel))
        vel_magnitude = np.linalg.norm(vel, axis=-1)

    # For 2D animations
    if ndim == 2:
        # make animation
        fig, ax = plt.subplots()

        def animate(i):
            fig.clear()
            ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
            ax.set_xlim(boundaries[0][0], boundaries[0][1])
            ax.set_ylim(boundaries[1][0], boundaries[1][1])
            ax.scatter(positions[i][:, 0], positions[i][:, 1], s=1)
            ax.grid(True, which='both')

    # For 3D animations
    if ndim == 3:
        print("Creating 3D animation...")
        # make animation
        fig = plt.figure(figsize=(10, 8))
        
        # Animation function
        def animate(i):
            i_frame = min(i, len(positions)-1)  # Ensure we don't go out of bounds
            print(f"Rendering frame {i_frame+1}/{len(positions)}")
            fig.clear()

            # Calculate proper index for velocity if using color
            if colorful:
                cmap = plt.cm.viridis
                vmax = np.ndarray.flatten(vel_magnitude).max()
                vmin = np.ndarray.flatten(vel_magnitude).min()
                sampled_value = vel_magnitude[i_frame]

            # Subsample particles if there are too many
            max_display = 100000
            if positions[i_frame].shape[0] > max_display:
                indices = np.random.choice(
                    positions[i_frame].shape[0], 
                    max_display, 
                    replace=False
                )
                pos_to_plot = positions[i_frame][indices]
                if colorful:
                    sampled_value = sampled_value[indices]
            else:
                pos_to_plot = positions[i_frame]

            if follow_taichi_coord:
                # Note: z and y is interchanged to match taichi coordinate convention.
                ax = fig.add_subplot(projection='3d', autoscale_on=False)
                ax.set_xlim(boundaries[0][0], boundaries[0][1])
                ax.set_ylim(boundaries[2][0], boundaries[2][1])
                ax.set_zlim(boundaries[1][0], boundaries[1][1])
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_zlabel("y")
                # ax.invert_zaxis()
                
                if colorful:
                    trj = ax.scatter(pos_to_plot[:, 0], pos_to_plot[:, 2], pos_to_plot[:, 1],
                                     c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=1.5, alpha=0.7)
                    fig.colorbar(trj)
                else:
                    ax.scatter(pos_to_plot[:, 0], pos_to_plot[:, 2], pos_to_plot[:, 1],
                               s=1.5, alpha=0.7, c='blue')
                
                # Set box aspect for better visualization
                ax.set_box_aspect(
                    aspect=(
                        float(boundaries[0][1]) - float(boundaries[0][0]),
                        float(boundaries[2][1]) - float(boundaries[2][0]),
                        float(boundaries[1][1]) - float(boundaries[1][0])
                    )
                )
                
                # Rotate view for better 3D perception
                ax.view_init(elev=20., azim=i*0.5)
                ax.grid(True, which='both')
            else:
                # Note: boundaries should still be permuted
                ax = fig.add_subplot(projection='3d', autoscale_on=False)
                ax.set_xlim(boundaries[0][0], boundaries[0][1])
                ax.set_ylim(boundaries[1][0], boundaries[1][1])
                ax.set_zlim(boundaries[2][0], boundaries[2][1])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.invert_zaxis()
                
                if colorful:
                    trj = ax.scatter(pos_to_plot[:, 0], pos_to_plot[:, 1], pos_to_plot[:, 2],
                                     c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=1.5, alpha=0.7)
                    fig.colorbar(trj)
                else:
                    ax.scatter(pos_to_plot[:, 0], pos_to_plot[:, 1], pos_to_plot[:, 2],
                               s=1.5, alpha=0.7, c='blue')
                
                # Set box aspect
                ax.set_box_aspect(
                    aspect=(
                        float(boundaries[0][1]) - float(boundaries[0][0]),
                        float(boundaries[1][1]) - float(boundaries[1][0]),
                        float(boundaries[2][1]) - float(boundaries[2][0])
                    )
                )
                
                ax.view_init(elev=20., azim=i * 0.5)
                ax.grid(True, which='both')

    # Calculate number of frames based on data size
    max_frames = 30
    num_frames = min(max_frames, len(positions))  # Limit to 1000 frames to keep file size manageable
    frame_indices = np.linspace(0, len(positions)-1, num_frames, dtype=int)
    
    # Create animation
    print(f"Creating animation with {num_frames} frames...")
    ani = animation.FuncAnimation(
        fig, animate, frames=frame_indices, interval=100, blit=False)

    # Try to save as GIF first
    try:
        print(f"Saving animation to: {path}/{save_name}.gif")
        ani.save(f'{path}/{save_name}.gif', dpi=80, fps=10, writer='pillow')
        print(f"Animation saved to: {path}/{save_name}.gif")
    except Exception as gif_err:
        print(f"Error saving GIF: {gif_err}")
        
        # Try to save as MP4 if GIF fails
        try:
            print(f"Trying to save as MP4 instead...")
            mp4_path = f'{path}/{save_name}.mp4'
            ani.save(
                mp4_path, 
                writer='ffmpeg', 
                fps=10, 
                dpi=100)#,
                #fargs=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
            #)
            print(f"Animation saved to: {mp4_path}")
        except Exception as mp4_err:
            print(f"Error saving MP4: {mp4_err}")
            raise
    
    plt.close(fig)
    return True

def animation_from_npy(npy_path, save_dir, boundaries=None, timestep_stride=5, max_frames=500, 
                       follow_taichi_coord=True, colorful=True):
    """
    Create animation from a .npy file containing simulation positions
    
    Args:
        npy_path (str): Path to the .npy file
        save_dir (str): Directory to save the animation
        boundaries (list): Simulation boundaries [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        timestep_stride (int): How many timesteps to skip between frames
        max_frames (int): Maximum number of frames to include in animation
        follow_taichi_coord (bool): Whether to follow Taichi coordinate convention
        colorful (bool): Whether to color particles by velocity
    
    Returns:
        bool: True if animation was created successfully, False otherwise
    """
    try:
        print(f"Creating animation from NPY file: {npy_path}")
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        
        # Load positions from NPY
        print(f"Loading positions from {npy_path}")
        positions = np.load(npy_path)
        
        # Determine number of dimensions
        ndim = positions.shape[-1]
        
        # Calculate velocity magnitudes for coloring if requested
        if colorful:
            initial_vel = np.zeros(positions[0].shape)
            initial_vel = initial_vel.reshape((1, initial_vel.shape[0], initial_vel.shape[1]))
            vel = positions[1:] - positions[:-1]
            vel = np.concatenate((initial_vel, vel))
            vel_magnitude = np.linalg.norm(vel, axis=-1)
            cmap = plt.cm.viridis
            vmax = np.ndarray.flatten(vel_magnitude).max()
            vmin = np.ndarray.flatten(vel_magnitude).min()
        
        # Set default boundaries if not provided
        if boundaries is None:
            # Calculate boundaries from positions
            mins = positions.min(axis=(0, 1))
            maxs = positions.max(axis=(0, 1))
            
            # Add some padding
            padding = (maxs - mins) * 0.1
            mins -= padding
            maxs += padding
            
            boundaries = [[mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]]]
            print(f"Using calculated boundaries: {boundaries}")
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        # Calculate stride to limit number of frames
        total_frames = len(positions)
        if total_frames > max_frames:
            adjusted_stride = max(1, total_frames // max_frames)
            stride = max(timestep_stride, adjusted_stride)
        else:
            stride = timestep_stride
            
        print(f"Animation will use {len(positions[::stride])} frames out of {total_frames} total frames")
        
        # Animation function
        def animate(frame_idx):
            frame = frame_idx * stride
            if frame >= len(positions):
                frame = len(positions) - 1
                
            fig.clear()
            
            if ndim == 2:
                ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
                ax.set_xlim(boundaries[0][0], boundaries[0][1])
                ax.set_ylim(boundaries[1][0], boundaries[1][1])
                
                if colorful:
                    scatter = ax.scatter(positions[frame][:, 0], positions[frame][:, 1], 
                                         c=vel_magnitude[frame], cmap=cmap, 
                                         vmin=vmin, vmax=vmax, s=1.5)
                    fig.colorbar(scatter, label='Velocity magnitude')
                else:
                    ax.scatter(positions[frame][:, 0], positions[frame][:, 1], s=1.5)
                    
                ax.grid(True)
                ax.set_title(f'Frame {frame}')
                
            elif ndim == 3:
                # Subsample particles for performance if there are too many
                max_display = 300000  # Maximum number of particles to display
                if positions[frame].shape[0] > max_display:
                    indices = np.random.choice(
                        positions[frame].shape[0], 
                        max_display, 
                        replace=False
                    )
                    positions_to_plot = positions[frame][indices]
                    if colorful:
                        sampled_value = vel_magnitude[frame][indices]
                else:
                    positions_to_plot = positions[frame]
                    if colorful:
                        sampled_value = vel_magnitude[frame]
                
                if follow_taichi_coord:
                    # Note: z and y is interchanged to match taichi coordinate convention
                    ax = fig.add_subplot(111, projection='3d', autoscale_on=False)
                    ax.set_xlim(boundaries[0][0], boundaries[0][1])
                    ax.set_ylim(boundaries[2][0], boundaries[2][1])
                    ax.set_zlim(boundaries[1][0], boundaries[1][1])
                    ax.set_xlabel("x")
                    ax.set_ylabel("z")
                    ax.set_zlabel("y")
                    #ax.invert_zaxis()
                    
                    if colorful:
                        scatter = ax.scatter(
                            positions_to_plot[:, 0], 
                            positions_to_plot[:, 2], 
                            positions_to_plot[:, 1],
                            c=sampled_value, cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.7
                        )
                        fig.colorbar(scatter, label='Velocity magnitude')
                    else:
                        # Attempt to separate water and obstacle particles by position
                        # This is an approximation - water is typically higher than obstacles
                        midpoint_y = np.median(positions_to_plot[:, 1])
                        
                        # Plot obstacle particles (below median y)
                        mask_obstacle = positions_to_plot[:, 1] < midpoint_y
                        ax.scatter(
                            positions_to_plot[mask_obstacle, 0],
                            positions_to_plot[mask_obstacle, 2],
                            positions_to_plot[mask_obstacle, 1],
                            color='green', s=1.5, alpha=0.8, label='Obstacle'
                        )
                        
                        # Plot water particles (above median y)
                        mask_water = ~mask_obstacle
                        ax.scatter(
                            positions_to_plot[mask_water, 0],
                            positions_to_plot[mask_water, 2],
                            positions_to_plot[mask_water, 1],
                            color='blue', s=1.5, alpha=0.7, label='Water'
                        )
                        ax.legend(loc='upper right')
                    
                    # Set box aspect ratio for better visualization
                    ax.set_box_aspect(
                        aspect=(
                            float(boundaries[0][1]) - float(boundaries[0][0]),
                            float(boundaries[2][1]) - float(boundaries[2][0]),
                            float(boundaries[1][1]) - float(boundaries[1][0])
                        )
                    )
                    
                    # Rotate view slightly for each frame for better 3D perception
                    ax.view_init(elev=20., azim=frame_idx * 0.5)
                    
                else:
                    # Standard coordinate system
                    ax = fig.add_subplot(111, projection='3d', autoscale_on=False)
                    ax.set_xlim(boundaries[0][0], boundaries[0][1])
                    ax.set_ylim(boundaries[1][0], boundaries[1][1])
                    ax.set_zlim(boundaries[2][0], boundaries[2][1])
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    
                    if colorful:
                        scatter = ax.scatter(
                            positions_to_plot[:, 0], 
                            positions_to_plot[:, 1], 
                            positions_to_plot[:, 2],
                            c=sampled_value, cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.7
                        )
                        fig.colorbar(scatter, label='Velocity magnitude')
                    else:
                        # Attempt to separate water and obstacle particles by position
                        midpoint_y = np.median(positions_to_plot[:, 1])
                        
                        # Plot obstacle particles (below median y)
                        mask_obstacle = positions_to_plot[:, 1] < midpoint_y
                        ax.scatter(
                            positions_to_plot[mask_obstacle, 0],
                            positions_to_plot[mask_obstacle, 1],
                            positions_to_plot[mask_obstacle, 2],
                            color='green', s=1.5, alpha=0.8, label='Obstacle'
                        )
                        
                        # Plot water particles (above median y)
                        mask_water = ~mask_obstacle
                        ax.scatter(
                            positions_to_plot[mask_water, 0],
                            positions_to_plot[mask_water, 1],
                            positions_to_plot[mask_water, 2],
                            color='blue', s=1.5, alpha=0.7, label='Water'
                        )
                        ax.legend(loc='upper right')
                    
                    # Set box aspect ratio for better visualization
                    ax.set_box_aspect(
                        aspect=(
                            float(boundaries[0][1]) - float(boundaries[0][0]),
                            float(boundaries[1][1]) - float(boundaries[1][0]),
                            float(boundaries[2][1]) - float(boundaries[2][0])
                        )
                    )
                    
                    # Rotate view slightly for each frame for better 3D perception
                    ax.view_init(elev=20., azim=frame_idx * 0.5)
                
                ax.grid(True)
                ax.set_title(f'Frame {frame}')
            
            return fig,
        
        # Create animation
        frame_count = min(100, len(positions[::stride]))  # Limit to 30 frames for file size
        print(f"Creating animation with {frame_count} frames...")
        
        ani = animation.FuncAnimation(
            fig, 
            animate, 
            frames=frame_count,
            blit=True,
            interval=100
        )
        
        # Save as MP4
        mp4_path = os.path.join(save_dir, f"{base_name}.mp4")
        print(f"Saving animation to {mp4_path}")
        
       # ani.save(
       #     mp4_path, 
       #     writer='ffmpeg', 
       #     fps=10, 
       #     dpi=100)#,
        #    fargs=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
        #)
        
        # Also save as GIF for easier viewing
        gif_path = os.path.join(save_dir, f"{base_name}.gif")
        print(f"Saving GIF animation to {gif_path}")
        
        try:
            ani.save(
                gif_path,
                writer='pillow',
                fps=10,
                dpi=80
            )
            print(f"GIF animation saved to {gif_path}")
        except Exception as gif_e:
            print(f"Failed to save GIF animation: {gif_e}")
        
        plt.close(fig)
        print(f"Animation creation completed successfully")
        
        return True
        
    except Exception as e:
        print(f"Error creating animation from NPY: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_material_points(
        mpm_solver: engine.mpm_solver.MPMSolver,
        ndim,
        particles_to_add: Union[str, list],
        material: int,
        velocity: list
):
    # generate cube-shaped particles from the list defining the cube shape,
    #   e.g., for 2d case, cube = [x_min, y_min, len_x, len_y]
    if type(particles_to_add) is list or type(particles_to_add) is tuple:
        mpm_solver.add_cube(
            lower_corner=[
                particles_to_add[0], particles_to_add[1], particles_to_add[2]] if ndim == 3 else [particles_to_add[0], particles_to_add[1]],
            cube_size=[
                particles_to_add[3], particles_to_add[4], particles_to_add[5]] if ndim == 3 else [particles_to_add[2], particles_to_add[3]],
            material=material,
            velocity=velocity)
    # generate particles from user defined csv files containing particle coordinate
    elif type(particles_to_add) is str:
        particle_coords = read_particles(particles_to_add)
        if particle_coords.shape[-1] != ndim:
            raise ValueError(
                f"Particle file is {particle_coords.shape[-1]}d data, but sim space is {ndim}d")
        mpm_solver.add_particles(
            particles=particle_coords,
            material=material,
            velocity=velocity
        )
    else:
        raise ValueError("Wrong input type for particle gen")

def add_pc_material_points(
        mpm_solver: engine.mpm_solver.MPMSolver,
        ndim,
        particles_to_add: Union[str, list],
        material: int,
        velocity: list
):
    # generate surface shape particles from the list defining the cube shape,
    #   e.g., for 2d case, cube = [x_min, y_min, len_x, len_y]
    if type(particles_to_add) is list or type(particles_to_add) is tuple:
        mpm_solver.add_cube(
            lower_corner=[
                particles_to_add[0], particles_to_add[1], particles_to_add[2]] if ndim == 3 else [particles_to_add[0], particles_to_add[1]],
            cube_size=[
                particles_to_add[3], particles_to_add[4], particles_to_add[5]] if ndim == 3 else [particles_to_add[2], particles_to_add[3]],
            material=material,
            velocity=velocity)
    # generate particles from user defined csv files containing particle coordinate
    elif type(particles_to_add) is str:
        particle_coords = read_pc_particles(particles_to_add)
        if particle_coords.shape[-1] != ndim:
            raise ValueError(
                f"Particle file is {particle_coords.shape[-1]}d data, but sim space is {ndim}d")
        mpm_solver.add_particles(
            particles=particle_coords,
            material=material,
            velocity=velocity
        )
    else:
        raise ValueError("Wrong input type for particle gen")

def read_particles(path):
    df = pd.read_csv(path, header=1)
    return df.to_numpy()

def read_pc_particles(path):
    print(f"Reading point cloud from: {path}")  # Debugging print
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud file {path} is empty or cannot be read.")
    points = np.asarray(pcd.points)
    return points
