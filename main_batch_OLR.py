# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:56:02 2023

@author: ikaro
"""

import pandas as pd
import numpy as np 
from tkinter import Tk
import csv
import math
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats as sts
from data_handling import *
from maze_modeling_OLR import *
from extract_frame import *
from data_processing import *
from plot_functions import *
from data_analysis import *
import cv2
from gif_functions import *
import os
import json


# SET parameter values
conf_threshold = 0.95
std_threshold = 0.5
fps = 24
prev_vertex_position = np.zeros((8,2))

# CREATE A DATA FRAME TO ORGANIZE THE RSULTS FOR ALL THE TRIALS
trial_info = pd.DataFrame(columns=['ID','Group','Day', 'Distance', 'Av_speed'])

# STEP 1 --> SELECT THE MULTIPLE FILES
filename = select_file(multiple=True)

# STEP 2 --> LOOP FOR EACH FILE
for it in range(len(filename)):
            
    # STEP 3 --> LOAD FILE AND RECREATE THE MAZE
    # Load the file
    df = pd.read_hdf(filename[it])
    # Fix camera shaking
    new_df = df_fix_camera_shaking(df)
    # Get each hole position
    position_each_vertex = get_vertex_positions(df, prev_vertex_position=prev_vertex_position) 
    prev_vertex_position = position_each_vertex # UPDATE THE HOLE POSITION IN CASE OF AN ERROR
    # Get the maze centroid
    centroid_coords = centroid_inference(position_each_vertex)    
    # Get the maze coordinates in pixel
    maze_info_pixel = maze_recreation(position_each_vertex, 10, 5, 60)
    # Recreate the maze quadrants
    quadrant_dict, quadrant_dict_list = maze_quadrants(maze_info_pixel, [], centroid_coords, position_each_vertex, plot_frame=False, title='Nose', show=True, recreate_maze=False)
    # Model the objects
    maze_info_pixel, maze_info_pixel_list = model_objects(df, conf_threshold, maze_info_pixel, centroid_coords, position_each_vertex)
    
    # STEP 3.1 --> GET THE NOSE AND HEAD COORD AND PROCESS IT
    # Get the coordinates for a specific body part
    nose_coord = df.xs('nose', level='bodyparts', axis=1).to_numpy()  
    # Fix coordinates inconsistencies based on confidence interval
    body_part_matrix_nose = fix_frames_confidence(nose_coord,conf_threshold)  
    ## OLD FUNCTION --> Fix coordinates inconsistencies based on (x,y) diff standard deviation
    # body_part_matrix_nose = fix_frames_diff(body_part_matrix_nose,std_threshold)
    # Get the coordinates for a specific body part
    head_coord = df.xs('head_centre', level='bodyparts', axis=1).to_numpy()  
    # Fix coordinates inconsistencies based on confidence interval
    body_part_matrix_head = fix_frames_confidence(head_coord,conf_threshold)  
    ## OLD FUNCTION --> Fix coordinates inconsistencies based on (x,y) diff standard deviation
    # body_part_matrix_nose = fix_frames_diff(body_part_matrix_nose,std_threshold)
    
    # STEP 3.2 --> GET THE TRIAL BEGINNING, END AND LATENCY
    # Define the trial beginning and end based on confidence interval
    #body_part_matrix_nose, beg, end = get_trial_beginning_end_all_bp(body_part_matrix_nose, df, 0.95)
    # OLD FUNCTION -->> body_part_matrix_nose = get_trial_beginning_end(body_part_matrix_nose, 0.95)
    # Get trial latency
    #latency = get_trial_latency(body_part_matrix_nose, fps=30)      
    
    # STEP 4 --> CREATE A CODE FOR THE NOSE POSITION ON MAZE
    # Get the nose position on the maze
    bp_pos_on_maze = get_bp_position_on_maze_OLR(body_part_matrix_nose, maze_info_pixel, centroid_coords, position_each_vertex, fps=fps)
    # Get the head angle
    head_angle = get_head_angle(body_part_matrix_nose,body_part_matrix_head)
    # Get the object exploration vector (throughout time)
    obj_exploration, obj_exp_parameters = get_obj_exploration(bp_pos_on_maze,head_angle, maze_info_pixel, body_part_matrix_nose, body_part_matrix_head, centroid_coords, position_each_vertex, fps=fps)
    ratio_1_total = obj_exp_parameters['ratio_1_total']
    ratio_2_total = obj_exp_parameters['ratio_2_total']
    time_obj_1 = obj_exp_parameters['time_obj_1']
    time_obj_2 = obj_exp_parameters['time_obj_2']

    # Filter the nose position
    bp_pos_on_maze_filtered = filter_bp_pos_on_maze(bp_pos_on_maze, method_used='complete', win=fps)
    
    # STEP 5 --> GET PRIMARY AND SECUNDARY ERRORS and STRATEGY USED
    p_errors, s_errors = get_p_s_errors(bp_pos_on_maze_filtered,target=1)[0:2]
    strategy = get_the_strategy(bp_pos_on_maze_filtered, target=1)
    
    # STEP 6.1 --> GET THE BODY CENTRE COORD AND PROCESS IT
    # Get the coordinates for a specific body part
    body_centre_coord = df.xs('body_centre', level='bodyparts', axis=1).to_numpy()  
    # Fix coordinates inconsistencies based on confidence interval
    body_part_matrix_body_centre = fix_frames_confidence(body_centre_coord,conf_threshold)
    
    # # UPDATE the beginning and end for the body_centre as well
    # body_part_matrix_body_centre, beg, end = get_trial_beginning_end_all_bp(body_part_matrix_body_centre, df, 0.95)
    
    # STEP 7 --> GET THE TOTAL DISTANCE, INSTANT SPEED AND AVERAGE SPEED
    total_distance = get_distance(body_part_matrix_body_centre, maze_info_pixel)[1]
    inst_speed, inst_speed_entire, av_speed = get_inst_speed(body_part_matrix_body_centre, maze_info_pixel, time_win=10, fps=fps)
        
    # STEP 7.1 --> BODY PART POSITION ON QUADRANT
    bp_pos_on_quadrant = get_bp_position_on_quadrant_OLR(body_part_matrix_body_centre, quadrant_dict, fps=fps)

      # STEP 7.2 --> RATIO (TIME ON TARGET/ TIME ON OTHER QUADRANTS)
    ratio_target_others, time_on_each_quadrant = get_time_on_target_quadrant(bp_pos_on_quadrant, target=1, fps=fps)    
      
    # STEP 8 --> ORGANIZE DATA SINCE NOT EVERY SINGLE THING IS BAGUNÇA
    # Get the file name (only the base name)
    basename = list(os.path.basename(filename[it]))
    ID = basename[1]
    group = ''.join(basename[5:7])
    day = basename[3]
       
    
    ######### Create a data frame to append to the final dataframe
    data = pd.DataFrame([[ID, group, day, total_distance, av_speed, ratio_1_total, ratio_2_total, time_obj_1, time_obj_2]], 
                        columns = ['ID','Group','Day','Distance', 'Av_speed', 'ratio_1_total', 'ratio_2_total', 'time_obj_1', 'time_obj_2']) 
    # makes index continuous
    trial_info = pd.concat([trial_info, data], ignore_index = True)  
    
    # Trial temporal series
    trial_temp_series = dict({'bp_pos_on_maze':bp_pos_on_maze.tolist(), 
                              'bp_pos_on_maze_filtered': bp_pos_on_maze_filtered.tolist(),
                              'bp_pos_on_quadrant':bp_pos_on_quadrant.tolist(),
                              'body_part_matrix_body_centre':body_part_matrix_body_centre.tolist(),
                              'body_part_matrix_nose':body_part_matrix_nose.tolist(),
                              'body_part_matrix_head':body_part_matrix_head.tolist(),
                              'position_each_vertex':position_each_vertex.tolist(),
                              'centroid_coords':centroid_coords.tolist(),
                              'maze_info_pixel':maze_info_pixel_list,
                              'quadrant_dict_list':quadrant_dict_list,
                              'total_distance':total_distance.tolist(),
                              'inst_speed':inst_speed.tolist(),
                              'inst_speed_entire':inst_speed_entire.tolist(),
                              'time_on_each_quadrant':time_on_each_quadrant.tolist(),
                              'obj_exp_parameters':obj_exp_parameters})
    
    # Get the name to save a file
    save_filename = os.path.dirname(filename[it])+'/'+''.join(basename[0:7])+'.txt'
    # Save temporal series as a json file
    with open(save_filename, "w", encoding='utf-8') as fp:
        json.dump(trial_temp_series, fp, indent=4)  # encode dict into JSON
        
    ############
   
    # STEP 9 --> PLOT WITH THE MAIN INFORMATIONS
    save_filename_plot = os.path.dirname(filename[it])+'/'+''.join(basename[0:7])
    big_plot_OLR(body_part_matrix_nose, body_part_matrix_body_centre, obj_exploration, centroid_coords, position_each_vertex, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, save_filename_plot, trial_data=data, show=True, fps=fps)
     
    # Lil' print to inform to the user that this specific trial has been analysed
    print(str(it)+' - '+''.join(basename[0:8]) + str(': OK!'))
    
# STEP 11 --> Save the final dataframe as json
save_filename = os.path.dirname(filename[it])+'/'+'Final_results'+'.h5'
trial_info.to_hdf(save_filename, key='trial_info', mode='w')  

trial_info = pd.read_hdf(save_filename, key='trial_info')  