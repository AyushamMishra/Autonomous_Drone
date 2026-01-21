import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

input_dem = r"D:\Autonomous_Drone\Search_map\cdnh43w\cdnh43w.tif"
output_dem = r"D:\Autonomous_Drone\Search_map\cdnh43w\cdnh43w_utm.tif"

target_crs = "EPSG:32643"  # UTM Zone 43N

with rasterio.open(input_dem) as src:
    transform, width, height = calculate_default_transform(
        src.crs,
        target_crs,
        src.width,
        src.height,
        *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        "crs": target_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    with rasterio.open(output_dem, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

print("DEM successfully reprojected to UTM")
