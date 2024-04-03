import numpy as np
from make_inequality_constraints import from_vertices_to_constraints

def test_constraints():
    vertices = np.array([[0, 0], [0, 1], [1,1], [1,0],[0,0]])
    coeffs=from_vertices_to_constraints(vertices, False) #True if plot needed
    assert coeffs==[([-1, 0], 0), ([0,1], 1), ([1,0], 1), ([0,-1],0)]

    vertices = np.array([[0, 0], [0, 5], [5, 0], [0, 0]])
    coeffs=from_vertices_to_constraints(vertices, False)
    assert coeffs==[([-5, 0], 0), ([5, 5], 25), ([0, -5], 0)]

    vertices=np.load("u_shaped_polygon.npy", allow_pickle=True)
    vertices=np.vstack([vertices, vertices[0]]) # connect polygon
    coeffs=from_vertices_to_constraints(vertices, False)
    assert coeffs==[([-281.0, -30.0], -56722.0), ([19.0, 542.0], 256002.0), ([272.0, 37.0], 205344.0), ([-181.0, -168.0], -161879.0), ([-60.0, -138.0], -83046.0), ([103.0, -170.0], -27115.0), ([128.0, -73.0], 9718.0)]


