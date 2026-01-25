# Projection script: converts lat, lon in x,y format
from pyproj import Transformer
class projection:
    def __init__(self,dem_crs):
        self.to_map=Transformer.from_crs(
            "EPSG:4326",dem_crs,always_xy=True
        )
    def lat_lon_to_xy(self,lat,lon):
        return self.to_map.transform(lon,lat)