import rasterio
import numpy as np
import plotly.graph_objects as go
from rasterio.warp import transform_bounds

# Чтение данных DEM из файла GeoTIFF
with rasterio.open("deterministic_algorithms/hilly_dem.dem") as src:
    dem_data = src.read(1)  # Читаем первый канал (высота)
    transform = src.transform
    crs = src.crs  # Получаем систему координат

# Преобразование границ в метрические координаты (WGS 84)
minx, miny, maxx, maxy = transform_bounds(crs, "+init=epsg:4326", *src.bounds)

# Создание координатной сетки в метрах
width = dem_data.shape[1]
height = dem_data.shape[0]
x_coords = np.linspace(minx, maxx, width)
y_coords = np.linspace(miny, maxy, height)
x_coords, y_coords = np.meshgrid(x_coords, y_coords)

# Создание 3D-поверхности для Plotly
fig = go.Figure(data=[go.Surface(
    x=x_coords, 
    y=y_coords, 
    z=dem_data, 
    colorscale='Viridis', 
    cmin=-90,cmax=50)])

fig.update_layout(
    title='3d График поверхности рельефа .DEM',
    autosize=False,
    scene=dict(
        xaxis_title='X (метры)',
        yaxis_title='Y (метры)',
        zaxis_title='Высота',
        zaxis=dict(range=[-20, 500])
    )
)

# Запуск интерактивного отображения
fig.show()

