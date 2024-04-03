import os
import matplotlib.pyplot as plt
import polygon
import pygame

current_directory=os.path.dirname(__file__)
import geopandas as gpd
from shapely.geometry import Polygon
import polygon
import matplotlib.pyplot as plt
import vmath

# Create a rectangular polygon
res = polygon.Polygon([vmath.Vector([0, 0]), vmath.Vector([0, 5]), vmath.Vector([10, 5]), vmath.Vector([5, 2]), vmath.Vector([10, 0])])
# rectangle_gdf=gpd.GeoDataFrame(geometry=[rectangle])



# # Plot the polygons for visualization
# ax = rectangle_gdf.plot(color='blue', alpha=0.5, figsize=(8, 8))
# #triangle_gdf.plot(ax=ax, color='red', alpha=0.5)
# #cut_rectangle_gdf.plot(ax=ax, color='green', alpha=0.5)
# ax.set_title('Rectangle with Cutout Triangle')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.legend(['Rectangle', 'Triangle', 'Cut Rectangle'])
# ax.set_aspect('equal', adjustable='box')

# # Display the plot in a window
# plt.show()

from run import displaying_polygon

if __name__=="__main__":
    cpg = polygon.ConvexPolygonsGroup(polygon.SimplePolygon(res))
    if displaying_polygon(res, cpg) == pygame.QUIT:
        print("Done")