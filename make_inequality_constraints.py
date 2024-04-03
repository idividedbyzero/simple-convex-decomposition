
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.optimize import LinearConstraint, minimize, OptimizeResult
from run import making_polygon
import polygon
import polygontransform

def from_vertices_to_constraints(vertices: np.ndarray, show: bool=True)->list[tuple]: 
    """Generate vectors a_i, b_i for inequality constraints <a_i,x><=b, simply form the vertices of a convex polygon

    Args:
        vertices (np.ndarray): Vertices of the convex polygon of shape Nx2
        show (bool, optional): Wether to show a plot of the situation. Defaults to True.

    Returns:
        list[tuple]: format ([...], .)=(a_i, b_i)
    """
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

def project_polygon(x: np.ndarray, poly: polygon.Polygon, show:bool=True)->np.ndarray:
    """Returns a projection of x onto some polygon.

    Args:
        x (np.ndarray): Vector to be projected
        poly (polygon.Polygon): Polygon to project onto
        show (bool, optional): Wether to show a plot of the situation. Defaults to True.

    Returns:
        np.ndarray: A vector on the polygon closest to the point x
    """
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
    project_polygon(x, res)

    