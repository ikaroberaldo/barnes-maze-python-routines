# Set of functions regarding minor calculations

import math

def isInside(circle_x, circle_y, rad, x, y):
     
    # Compare radius of circle
    # with distance of its center
    # from given point
    if ((x - circle_x) * (x - circle_x) +
        (y - circle_y) * (y - circle_y) <= rad * rad):
        return True;
    else:
        return False;

def mov_average(arr, window_size):
    # Program to calculate moving average 
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:

        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return moving_averages


# A utility function to calculate area
# of triangle formed by (x1, y1),
# (x2, y2) and (x3, y3)
 
def area(x1, y1, x2, y2, x3, y3):
 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)
 
 
# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside_triangle(triangle_matrix, x, y):
    
    x1 = triangle_matrix[0,0]
    y1 = triangle_matrix[0,1]
    x2 = triangle_matrix[1,0]
    y2 = triangle_matrix[1,1]
    x3 = triangle_matrix[2,0]
    y3 = triangle_matrix[2,1]
 
    # Calculate area of triangle ABC
    A = area (x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = area (x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = area (x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = area (x1, y1, x2, y2, x, y)
    
    soma = A1 + A2 + A3
     
    # Check if sum of A1, A2 and A3
    # is same as A
    if(A == A1 + A2 + A3):
        return True
    else:
        return False
    

# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside_triangle_dif(triangle_matrix, x, y):
    
    x1 = triangle_matrix[0,0]
    y1 = triangle_matrix[0,1]
    x2 = triangle_matrix[1,0]
    y2 = triangle_matrix[1,1]
    x3 = triangle_matrix[2,0]
    y3 = triangle_matrix[2,1]
 
    # Calculate area of triangle ABC
    A = area (x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = area (x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = area (x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = area (x1, y1, x2, y2, x, y)
    
    return abs((A1 + A2 + A3)-A)
     
  
 
