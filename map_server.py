import numpy as np
import rasterio 
from rasterio.transform import rowcol
import DEM_reprojection
from Projection import projection


class Map_Server:
    ##  DEM matching
    
    dem_path=DEM_reprojection.output_dem
    def __init__(self,dem_path):
        self.dataset=rasterio.open(dem_path)
        self.dem=self.dataset.read(1,masked=True)
        self.transform=self.dataset.transform
        self.crs=self.dataset.crs
        self.height,self.width=self.dem.shape
        self.bounds=self.dataset.bounds

        print(f"DEM loaded with shape: {self.dem.shape},CRS: {self.crs}")

        if not self.crs.is_projected:
            raise RuntimeError(
                f"DEM CRS {self.crs} is not projected. PF requires UTM / metric DEM."
            )

        ## MAP Resolution(meters per pixel)
        self.res_x=abs(self.transform.a)
        self.res_y=abs(self.transform.e)
        print(f"Bounds: {self.bounds}")
        print(f"Resolution: {self.res_x} m/pixel (x), {self.res_y} m/pixel (y)")
     

     ## Checking bounds 

    def in_bounds(self,x,y):
        return(
            self.bounds.left<=x<=self.bounds.right and
            self.bounds.bottom<=y<=self.bounds.top
        )

    # Elevation from DEM at given x,y

    def get_elevation(self,x,y):
        if not self.in_bounds(x,y):
            return np.nan  # out of bounds
        row,col=rowcol(self.transform,x,y)

        if row<0 or row>=self.height or col<0 or col>=self.width:
            return np.nan  # out of bounds
        
        z=self.dem[row,col]
        if np.ma.is_masked(z):
            return np.nan # no data value
        return float(z)
    
    def get_slope(self, x, y, yaw, step=1.0):
          
        ##   Returns terrain slope (dz/ds) along heading direction
    
        z1 = self.get_elevation(x, y)
        if np.isnan(z1):
            return np.nan

        x2 = x + step * np.cos(yaw)
        y2 = y + step * np.sin(yaw)
        z2 = self.get_elevation(x2, y2)
        if np.isnan(z2):
            return np.nan

        return (z2 - z1) / step

    
    def close(self):
        self.dataset.close()

    


    
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
    from DEM_reprojection import output_dem
    start_lat = 28.00
    start_lon = 76.00

    map_server = Map_Server(output_dem)
    proj = projection(map_server.crs)

    x0, y0 = proj.lat_lon_to_xy(start_lat, start_lon)
    estimator = StateEstimator(x0, y0)

    #Fake VIO loop
    for step in range(10):
        dx, dy, dyaw = 1.0, 0.2, 0.01  # meters, meters, radians
        x, y, yaw = estimator.update_from_vio(dx, dy, dyaw)

        z = map_server.get_elevation(x, y)
        slope=map_server.get_slope(x,y,yaw,step=1.0)
        print(f"Step {step}: x={x:.2f}, y={y:.2f}, z={z},slope={slope}")
