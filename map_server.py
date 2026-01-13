import numpy as np
import rasterio 
from rasterio.transform import rowcol

class Map_Server:
    ##  DEM matching
    def __init__(self,dem_path=r"D:\Autonomous_Drone\Search_map\cdnh43w\cdnh43w.tif"):
        self.dataset=rasterio.open(dem_path)
        self.dem=self.dataset.read(1,masked=True)
        self.transform=self.dataset.transform
        self.crs=self.dataset.crs
        self.height,self.width=self.dem.shape

        print(f"DEM loaded with shape: {self.dem.shape}, CRS: {self.crs}")

    def get_elevation(self,x,y):
        row,col=rowcol(self.transform,x,y)

        if row<0 or row>=self.height or col<0 or col>=self.width:
            return None  # out of bounds
        
        z=self.dem[row,col]
        if np.ma.is_masked(z):
            return None # no data value
        return float(z)
    
    def close(self):
        self.dataset.close()

# Projection script: converts lat, lon in x,y format
from pyproj import Transformer
class projection:
    def __init__(self,dem_crs):
        self.to_map=Transformer.from_crs(
            "EPSG:4326",dem_crs,always_xy=True
        )
    def lat_lon_to_xy(self,lat,lon):
        return self.to_map.transform(lon,lat)
    
# State estimator script
class StateEstimator:
    def __init__(self, x0, y0, yaw0=0.0):
        self.x = x0
        self.y = y0
        self.yaw = yaw0

    def update_from_vio(self, dx, dy, dyaw):
        # Rotate VIO motion into map frame
        cos_y = np.cos(self.yaw)
        sin_y = np.sin(self.yaw)

        self.x += cos_y * dx - sin_y * dy
        self.y += sin_y * dx + cos_y * dy
        self.yaw += dyaw

        return self.x, self.y,self.yaw


## Simulation script 

if __name__ == "__main__":

    #Initialization
    dem_path = r"D:\Autonomous_Drone\Search_map\cdnh43w\cdnh43w.tif"
    start_lat = 28.00
    start_lon = 76.00

    map_server = Map_Server(dem_path)
    proj = projection(map_server.crs)

    x0, y0 = proj.lat_lon_to_xy(start_lat, start_lon)
    estimator = StateEstimator(x0, y0)

    #Fake VIO loop
    for step in range(10):
        dx, dy, dyaw = 1.0, 0.2, 0.01  # meters, meters, radians
        x, y, yaw = estimator.update_from_vio(dx, dy, dyaw)

        z = map_server.get_elevation(x, y)
        print(f"Step {step}: x={x:.2f}, y={y:.2f}, z={z}")
