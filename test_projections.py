import numpy as np
from make_inequality_constraints import from_vertices_to_constraints, project_convex

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


def test_convex_projection():
    x=np.array([10,10])
    vertices = np.array([[0,0], [0, 5], [5, 0], [0,0]])
    coeffs=from_vertices_to_constraints(vertices, False)
    sol=project_convex(x, coeffs)
    assert np.isclose(np.linalg.norm(sol.x - np.array([2.5,2.5])), 0)

    x=np.array([10,0])
    vertices = np.array([[0,0], [0, 5], [5, 0], [0,0]])
    coeffs=from_vertices_to_constraints(vertices, False)
    sol=project_convex(x, coeffs)
    assert np.isclose(np.linalg.norm(sol.x - np.array([5.0, 0.0])), 0)

    #plot if necessary
    # plt.figure(figsize=(6, 6))
    # plt.scatter(x[0], x[1])
    # plt.scatter(sol.x[0], sol.x[1])
    # plt.plot(vertices[:, 0], vertices[:, 1], color='blue')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Projection')
    # plt.grid(True)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()