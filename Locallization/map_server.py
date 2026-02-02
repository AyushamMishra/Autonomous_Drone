
import numpy as np
import rasterio 
from rasterio.transform import rowcol
from Locallization import DEM_reprojection
from Locallization.Projection import projection


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
        self.res_x=abs(self.transform.a)    # Resolution in x direction (Pixel width)
        self.res_y=abs(self.transform.e)    # Resolution in y direction (Pixel height)
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
    
    def get_slope(self, x, y, yaw, step=None):

        if step is None:
            step = max(self.res_x, self.res_y)
        
          
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
    
    def get_gradient(self,x,y):
        # Returns terrain gradient (dz/dx, dz/dy) at (x,y)

        # DEM resolution
        dx = abs(self.res_x)
        dy = abs(self.res_y)

       # Central difference approximation
        z_x1 = self.get_elevation(x + dx, y)
        z_x2 = self.get_elevation(x - dx, y)
        z_y1 = self.get_elevation(x, y + dy)
        z_y2 = self.get_elevation(x, y - dy)

        if(
            np.isnan(z_x1) or np.isnan(z_x2) or
            np.isnan(z_y1) or np.isnan(z_y2)

        ):
            return (np.nan,np.nan)
        
        dz_dx=(z_x1-z_x2)/(2*dx)
        dz_dy=(z_y1-z_y2)/(2*dy)

        return (dz_dx,dz_dy)
    

    def get_aligned_patch(self,x,y,yaw,length,spacing):
        """ Samples DEM elevationsalong the direction of motion and
            returns 1D array of elevations. """
        
        num_samples=int(length/spacing)
        patch=np.zeros(num_samples)

        for i in range (num_samples):

            s=-i*spacing          # Backward sampling along motion direction

            xi=x+s*np.cos(yaw)
            yi=y+s*np.sin(yaw)

            zi=self.get_elevation(xi,yi)
            if zi is None or np.isnan(zi):
                return None      # Invalid patch
            patch[i]=zi

        return patch
    
    def get_2D_terrain_patch(self,x,y,length,spacing,half_width_cell=2):
        """ Samples a 2D patch centered at (x,y)
            returns 2D array of elevations which are yaw aligned 
            Shape: [N_along,N_cross]  """
        
        N_along=int(length/spacing)
        N_cross=2*half_width_cell + 1

        patch_2D=np.full((N_along,N_cross),np.nan)    # Definindg 2D patch with NaNs

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        for i in range(N_along):
            s = -i * spacing   # backward in time (match altitude history)

            cx = x + s * cos_y
            cy = y + s * sin_y

            for j, n in enumerate(range(-half_width_cell, half_width_cell + 1)):
                nx = cx + n * spacing * (-sin_y)
                ny = cy + n * spacing * ( cos_y)

                z = self.get_elevation(nx, ny)
                patch_2D[i, j] = z

        return patch_2D
         
    

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

        dz_dx, dz_dy = map_server.get_gradient(x, y)
        print(f"grad_x={dz_dx:.4f}, grad_y={dz_dy:.4f}")

        z = map_server.get_elevation(x, y)
        slope=map_server.get_slope(x,y,yaw,step=1.0)
        print(f"Step {step}: x={x:.2f}, y={y:.2f}, z={z},slope={slope}")
