import numpy as np
from fusion import Track, PoseStamped

'''
A simple tracker which takes in 3D measurements, associates them to a respective track via stat and real dist, 
and updates the track state accordingly.

When a track matures, its latest state can be accessed.
When a track hasn't been updated for 5 seconds, it is stored and removed from active tracking.

Every measurement is a 3D numpy array: [x, y, z] (meters) with an associated measurement noise R (3x3 numpy array) and bounding box.
The bounding box is stored with the track and so is its history [width, height] in meters.
'''