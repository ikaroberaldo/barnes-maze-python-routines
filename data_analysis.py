# Set of functions for data analysis such as used strategy, animal speed, time spent on a certain part of the maze

from minors_functions import *
from matplotlib import pyplot as plt
import numpy as np
from color_functions import *

# Extract the animal position relative to the maze holes
def get_bp_position_on_maze(body_part_matrix, position_each_hole, maze_info_pixel, centroid_coords, fps=30):
    
    x = body_part_matrix[:,0]  # x coords
    y = body_part_matrix[:,1]  # y coords
    hole_radius = maze_info_pixel["hole_radius_pixels"]
    maze_radius = maze_info_pixel["maze_radius_pixels"]
    
    # Create a body part position vector (coded as 1 to 12 for holes, 0 for the maze and -1 when outside the maze)
    bp_pos_on_maze = np.zeros((len(x),))
    
    # Loop for each frame
    for i in range(len(x)):
        # Loop for each hole
        for ii in range(np.shape(position_each_hole)[0]):
            # Check if the body part is inside the specific hole
            if isInside(position_each_hole[ii,0], position_each_hole[ii,1], hole_radius, x[i], y[i]): 
                bp_pos_on_maze[i] = ii + 1  # Insert the index regarding the role (+1, since python indexing is quite differente from what I've learned in MATLAB)

    ## Check if the body part is outside the maze (ignore the parts which are already on holes)
    on_maze = np.where(bp_pos_on_maze == 0)
    for i in np.transpose(on_maze):
        if not(isInside(centroid_coords[0], centroid_coords[1], maze_radius, x[i], y[i])): # If the body part is inside the maze
            bp_pos_on_maze[i] = -1  # insert the code 1
    
    return bp_pos_on_maze

# Extract the number of primary errors on Barnes Maze (default target hole = 1)
def get_p_s_errors(bp_pos_on_maze,target=1):
    # Get the entries and exits from holes
    entrie_exit = np.array(np.where(np.diff(bp_pos_on_maze) != 0))
    
    # Compute the total number of errors
    errors = (bp_pos_on_maze[np.array(entrie_exit)+1] != target) & (bp_pos_on_maze[np.array(entrie_exit)+1] != 0) & (bp_pos_on_maze[np.array(entrie_exit)+1] != -1)
    
    # Get the number of primary errors
    p_errors = len(np.unique(bp_pos_on_maze[np.array(entrie_exit[errors])+1]))
    # Get the number of secundary
    s_errors = len(bp_pos_on_maze[np.array(entrie_exit[errors])+1]) - p_errors  
    
    # Holes regarding the errors
    h_errors = bp_pos_on_maze[np.array(entrie_exit[errors])+1]
    
    # Indices regarding the errors
    i_errors = np.array(entrie_exit[errors])+1
    
        
    return p_errors, s_errors, h_errors, i_errors

def get_distance(body_part_matrix, maze_info_pixel):
    # Euclidean distance
    euc_distance_frame = np.array(np.sqrt(np.diff(body_part_matrix[:,0])**2 + np.diff(body_part_matrix[:,1])**2))
    # Convert to cm
    euc_distance_frame = euc_distance_frame/maze_info_pixel["pixelcm_ratio"]
    # Add a 0 for the first frame
    euc_distance_frame = np.append([0], euc_distance_frame, axis=0)
    # Get the total distance
    total_distance = np.sum(euc_distance_frame)

        
    return euc_distance_frame, total_distance

# Function to extract the animals instantaneous speed
def get_inst_speed(body_part_matrix, maze_info_pixel, time_win=1, fps=30):
    # Get the distance frame by frame (only the first output)
    euc_distance_frame = get_distance(body_part_matrix, maze_info_pixel)[0]
    # Time window in sec
    time_win_sec = time_win/fps
    # Calculate speed
    n_bins = np.int64(np.floor(euc_distance_frame.shape[0]/time_win))
    inst_speed = np.zeros((n_bins,))
    inst_speed_entire = np.zeros((len(euc_distance_frame),))
    
  
    for i in range(len(inst_speed)):
        # Instantaneous speed based on the time window
        a = time_win*i             # First element
        b = time_win*(i+1) - 1     # Last element
        inst_speed[i] = np.sum(euc_distance_frame[np.arange(a,b+1,1)])/time_win_sec    # isntantaneous speed
        inst_speed_entire[np.arange(a,b+1,1)] = np.sum(euc_distance_frame[np.arange(a,b+1,1)])/time_win_sec       # Instaneous speed, but filling every single frame value (even if it is repeated, important for plot)
           
    # Mean speed
    av_speed = np.mean(inst_speed)
        
    return inst_speed, inst_speed_entire, av_speed
        
    
# Define the strategy used by the animal
def get_the_strategy(bp_pos_on_maze, target=1):
    # First get the primary and secundary errors, including the hole numbers
    p_errors, s_errors, h_errors = get_p_s_errors(bp_pos_on_maze, target)[0:3]
    
    # If it doesn't fall to any of those conditions, classify the strategy as random
    strategy = 'random'
    
    # Direct/Spatial strategy
    if p_errors + s_errors < 3:
        # Check if holes were whithin 2 from target (only holes with difference 1, 3, 11 and 10)
        if not(((np.abs(h_errors-target) > 2) & (np.abs(h_errors-target) < 10)).all()):
            strategy = 'spatial'
        elif p_errors + s_errors <= 1: # One error far from the target or 0 errors
            strategy = 'spatial'
        else:
            strategy = 'random'
    
    # (serial/random)
    else:
        # Add the target role to the end of h_erros if it is serial
        h_errors = np.append([[h_errors]],[[target]])
        # Check if at least 75% of the exploration was serial (
        serial_mask = ((np.abs(np.diff(h_errors)) == 1) | (np.abs(np.diff(h_errors)) == 0) | (np.abs(np.diff(h_errors)) == 11).all())
        
        # If 50% of searchs were serial
        if np.array(np.where(serial_mask == True)).shape[1] >= len(h_errors)*0.75:
            strategy = 'serial'
        else:
            strategy = 'random'
    
    return strategy


# Function to get the trial latency
def get_trial_latency(body_part_matrix, fps=30):
    # get the number of frames
    n_frames = np.shape(body_part_matrix)[0]
    # latency
    latency = n_frames/fps
    
    return latency

# Extract the animal position relative to the maze quadrants
def get_bp_position_on_quadrant(body_part_matrix, quadrant_dict, fps=30):
    
    x = body_part_matrix[:,0]  # x coords
    y = body_part_matrix[:,1]  # y coords
    
    
    # Create a body part position vector (coded as 1 to 4 for each one of the quadrants)
    bp_pos_on_quadrant = np.zeros((len(x),))
    
    # Loop for each frame
    for i in range(len(x)):
        # Loop for each quadrant
        compare = np.zeros((4,))
        for ii in range(len(quadrant_dict)):
            # Check if the body part is positioned on the specific
            triangle = quadrant_dict['quadrant'+str(ii+1)]            
            
            # Get the diference between areas -> inference for animal distance
            compare[ii] = isInside_triangle_dif(triangle, x[i], y[i])
        
        # Get the index of the nearest quadrant (considering the animal's bp)       
        bp_pos_on_quadrant[i] = np.argmin(compare) + 1 # Insert the index regarding the role (+1, since python indexing is quite differente from what I've learned in MATLAB)

    return bp_pos_on_quadrant

# Get the time spent on each quadrant and calculate the ratio time on the target quadrant / time on the other quadrants
def get_time_on_target_quadrant(bp_pos_on_quadrant, target=1, fps=30):
    # Create an zeros array
    time_on_each_quadrant = np.zeros(4,)
    # Get the time on each quadrant
    time_on_each_quadrant[0] = np.count_nonzero(bp_pos_on_quadrant == 1)/fps
    time_on_each_quadrant[1] = np.count_nonzero(bp_pos_on_quadrant == 2)/fps
    time_on_each_quadrant[2] = np.count_nonzero(bp_pos_on_quadrant == 3)/fps
    time_on_each_quadrant[3] = np.count_nonzero(bp_pos_on_quadrant == 4)/fps
    
    # Calculate the ratio target/others
    ratio_target_others = time_on_each_quadrant[target-1]/np.sum(time_on_each_quadrant)
    
    return ratio_target_others, time_on_each_quadrant
    
    
    