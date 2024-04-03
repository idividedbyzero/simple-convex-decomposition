
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.optimize import LinearConstraint, minimize, OptimizeResult
import pygame
from run import making_polygon
import polygon
import polygontransform

import itertools

def from_vertices_to_constraints(vertices: np.ndarray, show: bool=True)->list[tuple]: 
    # Initialize lists to store coefficients
    coefficients = []

    # Iterate over each polygon
    #for polygon in polygons.geometry:
    # Extract vertices of the polygon

    # Calculate coefficients for each edge
    for i in range(len(vertices) - 1):
        x1, y1 = vertices[i]
        x2, y2 = vertices[i + 1]

        # Calculate edge vector
        edge_vector = [y1 - y2, x2 - x1]

        # Calculate coefficients
        a = edge_vector
        b = np.dot(edge_vector, [x1, y1])

        # Append coefficients to the list
        coefficients.append((a, b))

    # Print the coefficients
    print("The coefficients: <a_i, x><=b_i are given by:")
    for idx, (a, b) in enumerate(coefficients):
        print(f"a_{idx}: {a}, b_{idx}: {b}")
    if show: 
        # Plot the polygon
        plt.figure(figsize=(6, 6))
        plt.plot(vertices[:, 0], vertices[:, 1], '-o', color='blue')
        for i_ in range(len(vertices)-1):
            x_start, y_start = 1/2*(vertices[i_]+vertices[i_+1])
            dx, dy = coefficients[i_][0]
            length=np.linalg.norm(coefficients[i_][0])
                # Plotting the arrow
            plt.arrow(x_start, y_start, dx, dy,length_includes_head=True,
                        head_width=0.1*length, head_length=0.1*length, fc='green', ec='green')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polygon Plot')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    return coefficients

def nonempty_subsets(n:int)->List:
    # Generate all subsets of {0, 1, ..., n}
    all_subsets = itertools.chain.from_iterable(itertools.combinations(range(0, n), r) for r in range(n+1)) 
    nonempty_subsets = filter(lambda x: len(x) > 0, all_subsets)
    return [np.array(a) for a in list(nonempty_subsets)]

def get_G(a_s: List[tuple])->np.ndarray:
    #Obtain the G matrix in the paper after numbering (26)
    return np.dot(a_s, a_s.T)

def is_singular(matrix):
    if matrix.ndim == 1:  # Handle the case when matrix is a 1D array
        matrix = np.reshape(matrix, (1, 1))  # Reshape to a 1x1 matrix
    return np.linalg.det(matrix) == 0

def solve_linear_system(small_G, RHS):
    if small_G.ndim == 1:  # Handle the case when small_G is a 1D array
        small_G = np.reshape(small_G, (1, 1))  # Reshape to a 1x1 matrix
        RHS = np.reshape(RHS, (1,))  # Reshape RHS to a 1D array
    return np.linalg.solve(small_G, RHS)

def project_convex(x: np.ndarray, coeffs: List[tuple])->OptimizeResult:
    """Solves the quadtratic Programming min_v |x-v| s.t. <a_i,v><=b_i
    Args:
        x (np.ndarray): Vector to project
        coeffs (List[tuple]): List of tuples of the form (a_i, b_i) where a_i is a 2D vector and b_i a float

    Returns:
        OptimizeResult: Solution of scipy minimize. OptimizeResult.x contains the optimal vector.
    """
    a_s = np.array([_c[0] for _c in coeffs])
    b_s = np.array([_c[1] for _c in coeffs])
    lc=LinearConstraint(a_s, ub=b_s)
    loss=lambda v: 1/2*np.linalg.norm(x-v, ord=2)**2
    return minimize(loss, x, constraints=lc)
    #Using the paper: https://arxiv.org/pdf/1607.00102.pdf
    #TODO if necessary for banach spaces...still not working e.g for [-10, 1]
    # n=len(coeffs)
    # Delta=nonempty_subsets(n)
    # a_s = np.array([_c[0] for _c in coeffs])
    # b_s = np.array([_c[1] for _c in coeffs])
    # G=get_G(a_s)
    # for m_, I_ in enumerate(Delta):
    #     #P=len(I_)
    #     small_G=G[np.ix_(I_, I_)]
    #     if is_singular(small_G):  #np.linalg.det(matrix) == 0 should suffice
    #         continue
    #     RHS=np.einsum("i,ki->k", x, a_s[I_])-b_s[I_] #(P)
    #     mu=solve_linear_system(small_G, RHS) # handle one dimensional case
    #     if np.any(mu<=0):
    #         continue
    #     Ic=[i_ for i_ in range(n) if i_ not in I_] #I_ complement
    #     diff=x-np.einsum("i,ij->j", mu, a_s[I_])
    #     if np.all(np.einsum("i,qi->q", diff, a_s[Ic])<=0):
    #         return diff
    # raise Exception("There must be at least one index set, satisfying the equations")

def project_polygon(x: np.ndarray, poly: polygon.Polygon, show:bool=True)->np.ndarray:
    cpg = polygon.ConvexPolygonsGroup(polygon.SimplePolygon(poly))
    #TODO in order to make this work one requires, to check if the polygon were made in the correct order
    p = polygontransform.ConvexPolygonGroup(cpg).poly_array
    coords=[np.array([e.pos.to_float_list() for e in  p_.p.vector_array]) for p_ in p]
    distances=np.zeros(len(coords))
    sols=[]
    for c_, vertices in enumerate(coords):
        if not np.array_equal(vertices[0], vertices[-1]):
            vertices=np.vstack([vertices, vertices[0]])
        coeffs=from_vertices_to_constraints(vertices, False)
        sol=project_convex(x, coeffs)
        sols.append(sol.x)
        distances[c_]=np.linalg.norm(x-sol.x)
    projected=sols[np.argmin(distances)]
    if show:
        plt.figure()
        plt.scatter(x[0], x[1])
        plt.scatter(projected[0], projected[1])
        for vertices in coords:            
            plt.plot(vertices[:, 0], vertices[:, 1])
            plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Triangle Plot')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    return projected
    
    
    


if __name__=="__main__":
    # Define the polygon vertices
    #vertices = np.array([[0, 0], [0, 5], [10, 5], [5, 2], [10, 0], [0, 0]])
    x=np.array([0,0])    
    res = making_polygon()
    print(project_polygon(x, res))

    