# IMPORT RELEVANT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import os
import shutil as sh

"""
Note: Results of the experiment saved in folder named 'task1_results' in the same directory as the python script

""" 
# Loading Image Data for Task1 program
cdata = np.load("label_train00.npy").T

# Function for Constructing the triangulated mesh by surface diving in task 1
def surface_dividing(triag_surf, sag_plane):
    """
        Function for diving a triangulated surface into two parts using a specified scalar point on sagittal plane

        SYNTAX:
            x1, x2 = surface_dividing(triag_surf, sag_plane)
            left_surf, right_surf = surface_dividing(triag_surf, sag_plane)

        INPUTS:
            triag_surf - list of vertices and triangles. eg: triag_surf = [vertices, faces] - (datatype: list)
            sag_plane - scalar value representing sagittal plane - (datatype: float)

        OUTPUTS:
            Returns lists of left and right surfaces containing vertices and triangles respectively
            left_surf - List containing vertices and triangles on the left surface [list,list]- (datatype: list)
            right_surf - List containing vertices and triangles on the right surface [list,list]- (datatype: list)

    """   
    # Create vertices and triangle variables for triangulated surface inputs
    vertices = triag_surf[0]
    faces = triag_surf[1]
    
    # Create empty lists for vertices and triangles separated on left(l) by sagittal plane
    l_surf_vert = []
    l_surf_tria = []
    
    # Create empty lists for vertices and triangles separated on right(r) by sagittal plane
    r_surf_vert = []
    r_surf_tria = []

    # Sagittal plane - x axis: List comprehension for left and right vertices based on position of sagittal plane set by scalar value
    l_surf_vert = [tuple(v) for v in vertices if v[0] < sag_plane]
    r_surf_vert = [tuple(v) for v in vertices if v[0] >= sag_plane]

    # Find position of triangle based on the list of vertices on either side of sagittal plane
    for t in faces:
        
        # Extract vertices of the triangle
        vert_tria= [vertices[t[0]], vertices[t[1]], vertices[t[2]]]
    
        # list comprehensions to determine position of triangle along left and right of sagittal
        sag_left = sum(1 for v in vert_tria if v[0] < sag_plane)
        sag_right = sum(1 for v in vert_tria if v[0] >= sag_plane)

        # Add triangles on the left and right into the appropriate list
        if sag_left== 3:
            
            # List of vertex indices which make up the triangle on the left
            l_index = [l_surf_vert.index(tuple(vert_tria[i])) for i in range(len(vert_tria))]
            l_surf_tria.append(tuple(l_index))

        elif sag_right== 3:
            
            # List of vertex indices which make up the triangle on the right
            r_index = [r_surf_vert.index(tuple(vert_tria[i])) for i in range(len(vert_tria))]
            r_surf_tria.append(tuple(r_index))

        else:
            
            # Plane intersects a triangle. Causing it to split into two points 
            # Empty list for intersection points
            points = []
            
            # Find vertex indices on left and right
            l_index = []
            r_index = []

            # Determine intersection points between plane and edges of triangle
            for i in range(len(vert_tria)):
                
                vt1 = tuple(vert_tria[i])
                vt2 = tuple(vert_tria[(i + 1) % 3])
                
                # Condition to check if the plane in sagittal orientation intersects with triangle edges
                if (vt2[0] < sag_plane < vt1[0]) or (vt1[0] < sag_plane < vt2[0]):
                    # y=mx+c line equation where m gives the slope
                    k = (sag_plane-vt1[0]) / (vt2[0]-vt1[0])
                    sag_plane0 = vt1[1] + k * (vt2[1]-vt1[1])
                    sag_plane1 = vt1[2] + k * (vt2[2]-vt1[2])
                    #Tuple with the intersection points across vt1 and vt2
                    c = (sag_plane, sag_plane0, sag_plane1)
                    # Condition checking whether the intersection is not equal to the two vertices of triangle
                    if c not in (vt1,vt2):
                        points.append(c)

            # Append the points of intersections into the list of left and right surface vertices.  
            for i in points:

                # To ensure points are not already present in the list for left surface 
                if i not in l_surf_vert:
                    l_surf_vert.append(i)
                # To ensure points are not already present in the list for right surface
                if i not in r_surf_vert:
                    r_surf_vert.append(i)

            # Determine index on left and right of sagittal plane for each vertex of triangle
            l_index = [l_surf_vert.index(tuple(v)) for v in vert_tria if v[0] < sag_plane]
            r_index = [r_surf_vert.index(tuple(v)) for v in vert_tria if v[0] >= sag_plane]

            # Forming new triangles by adding vertices produced by intersection of plane
            #Triangles on Right Sagittal Plane
            if sag_right != 1:
                
                r_surf_tria.append((r_index[0], r_index[1], r_surf_vert.index(points[0])))
                r_surf_tria.append((r_index[0], r_index[1], r_surf_vert.index(points[1])))
                r_surf_tria.append((r_index[1], r_surf_vert.index(points[0]), r_surf_vert.index(points[1])))
                
            else:
                
                r_surf_tria.append((r_index[0], r_surf_vert.index(points[0]), r_surf_vert.index(points[1])))

            #Triangles on Left Sagittal Plane
            if sag_left!= 1:

                l_surf_tria.append((l_index[0], l_index[1], l_surf_vert.index(points[0])))
                l_surf_tria.append((l_index[0], l_index[1], l_surf_vert.index(points[1])))
                l_surf_tria.append((l_index[1], l_surf_vert.index(points[0]), l_surf_vert.index(points[1])))

            else:

                l_surf_tria.append((l_index[0], l_surf_vert.index(points[0]), l_surf_vert.index(points[1])))

    #List of two triangulated surfaces on left and right
    left_surf = [l_surf_vert, l_surf_tria]
    right_surf = [r_surf_vert, r_surf_tria]

    # Return the lists of left and right triangulated surfaces divided             
    return left_surf, right_surf


def create_dir(folder):
    """
        Function for creating a folder in directory with the same location where python script runs

        SYNTAX:
            create_dir('task1_results')

        INPUTS:
            folder - Name of the folder - (datatype: string)

        OUTPUTS:
            Creates a folder with the specified name for task

    """
    # Checks if the folder containing the same name exists or not   
    if os.path.exists(folder):
        sh.rmtree(folder)
        os.mkdir(folder)  
    else:
        # Makes the folder inside the directory
        os.mkdir(folder)  


def plot_triangles(vertices, faces, case):

    """
        Function for plotting triangles using vertices and faces

        SYNTAX:
            plot_triangles(vertices, faces, 1)

        INPUTS:
            vertices - Numpy array containing vertices - (datatype: numpy.ndarray)
            faces - Numpy array containing triangles - (datatype: numpy.ndarray)
            case - for selecting the specific sagittal plane, colormap and title of the file - (datatype:int)

        OUTPUTS:
            Saving the results of the output in PNG file format inside the 'task1_results' folder

    """   
    # Checking CASE 1, CASE 2, CASE 3
    if case == 1:
        sag_plane = 17.2
        cmap_inp = 'rainbow'
        title = 'case1'
    elif case == 2:
        sag_plane = 23.8
        cmap_inp = 'rainbow'
        title = 'case2'
    else:
        sag_plane = 44.7
        cmap_inp = 'rainbow'
        title = 'case3'
    
    #Calling surface_dividing() to extract left and right surface vertices and triangles
    surf_triangul = [vertices, faces]
    left_surf, right_surf = surface_dividing(surf_triangul, sag_plane)

    # Getting the values of left surface vertices and triangles
    left_vert = np.array(left_surf[0])
    left_tria = np.array(left_surf[1])

    # Getting the values of right surface vertices and triangles
    right_vert = np.array(right_surf[0])
    right_tria = np.array(right_surf[1])

    # Creating the figure and 3D axes for left surface
    fig = plt.figure(figsize= (10,10))
    p = fig.add_subplot(111, projection='3d')

    # Plot the triangle faces using left vertices
    p.plot_trisurf(left_vert[:,0], left_vert[:,1], left_vert[:,2], triangles=left_tria, cmap=cmap_inp)

    # Show plot
    # plt.show()
    # Save image in PNG file
    fig.savefig('task1_results/' + title + '_left.png')

    # Creating the figure and 3D axes for right surface
    fig = plt.figure(figsize= (10,10))
    p = fig.add_subplot(111, projection='3d')

    # Plot the triangle faces using right vertices
    p.plot_trisurf(right_vert[:,0], right_vert[:,1], right_vert[:,2], triangles=right_tria, cmap=cmap_inp)

    # Show plot
    # plt.show()
    # Save image in PNG file
    fig.savefig('task1_results/' + title + '_right.png')
    

#----------------------------------MAIN SCRIPT---------------------------------

# Applying Marching Cube to find the vertex coordinates in mm and triangle values representing segmented boundaries
vertices, faces, _, _ = sk.measure.marching_cubes(cdata, 0, spacing = (0.5, 0.5, 2), step_size=2)

# Create folder for savinf results
create_dir('task1_results')

# CASE 1 - Scalar value of Sagittal Plane in the middle of the 3D image
plot_triangles(vertices, faces, 1)

# CASE 2 - Scalar value of Sagittal Plane towards the left of the 3D image
plot_triangles(vertices, faces, 2)

# CASE 3 - Scalar value of Sagittal Plane towards the right of the 3D image
plot_triangles(vertices, faces, 3)
