import rasterio
import numpy as np
import plotly.graph_objects as go
from rasterio.warp import transform_bounds
from scipy.ndimage import label, binary_erosion, binary_dilation

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


# Создание 2D-графика для Plotly
fig = go.Figure(data=[go.Contour(
    x=x_coords[0],  # Используем первую строку x_coords
    y=y_coords[:, 0],  # Используем первый столбец y_coords
    z=dem_data,
    colorscale='Viridis',
    contours=dict(start=np.min(dem_data), end=np.max(dem_data), size=25),
    
)])

fig.update_layout(
    title='Тот-же график рельефа, но вид с верху',
    xaxis_title='X (метры)',
    yaxis_title='Y (метры)'
)

# Запуск интерактивного отображения
fig.show()
