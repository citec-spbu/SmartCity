import rasterio
import numpy as np
import plotly.graph_objects as go
from rasterio.warp import transform_bounds
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import disk
import matplotlib.pyplot as plt
import plotly.colors as pc

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


def find_flat_areas(dem, min_height=0, max_height_diff=1, min_radius=10, resolution=1):
    """Находит плоские участки на DEM с минимальной высотой и перепадом.
        Возвращает маску, где каждому участку соответствует уникальный номер

    Args:
        dem (np.array): Массив высот DEM.
        min_height (int): Минимальная высота для плоской области.
        max_height_diff (int): Максимальный перепад высот для плоской области.
        min_radius (int): Минимальный радиус плоской области в метрах.
        resolution (float): Размер пикселя в метрах

    Returns:
        np.array: Массив с метками плоских участков, 0 - не плоская область.
    """

    # Создаем бинарную маску, где True - плоские участки.
    is_flat = np.zeros_like(dem, dtype=bool)

    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            if dem[i, j] >= min_height:
                window = dem[max(0, i - 1):min(dem.shape[0], i + 2),
                            max(0, j - 1):min(dem.shape[1], j + 2)]
                if np.max(window) - np.min(window) <= max_height_diff:
                    is_flat[i, j] = True
    
    #Заполняем дыры
    is_flat = binary_fill_holes(is_flat)

    # Поиск связных областей (кластеров) плоских участков
    labeled_flat_areas, num_labels = label(is_flat)

    # Фильтруем по размеру и удаляем пересекающиеся области
    min_pixel_radius = int(min_radius / resolution)
    filtered_flat_areas = np.zeros_like(labeled_flat_areas)
    sizes = []
    labels_to_keep = []
    for label_num in range(1, num_labels + 1):
        area_mask = labeled_flat_areas == label_num
        size = np.sum(area_mask)

        # Проверка на размер и удаление пересечений
        if size >= np.pi * min_pixel_radius**2:
            is_overlapping = False
            for prev_label in labels_to_keep:
                prev_mask = labeled_flat_areas == prev_label
                overlap = np.sum(area_mask & prev_mask)

                if overlap > 0:  # Если пересечение найдено
                    is_overlapping = True
                    break
            if not is_overlapping:
                labels_to_keep.append(label_num)  # Сохраняем метку
                filtered_flat_areas += area_mask * label_num

    return filtered_flat_areas


# Находим все подходящие плоские участки
resolution_x = (maxx-minx)/width
resolution_y = (maxy-miny)/height
resolution = (resolution_x + resolution_y)/2
labeled_areas = find_flat_areas(dem_data, min_height=0, max_height_diff=1, min_radius=10, resolution = resolution)

# Создание 2D-графика для Plotly
fig = go.Figure()

fig.add_trace(go.Contour(
    x=x_coords[0],
    y=y_coords[:, 0],
    z=dem_data,
    colorscale='Viridis',
    contours=dict(start=np.min(dem_data), end=np.max(dem_data), size=25),
))

# Отображаем контуры каждого плоского участка
if np.max(labeled_areas) > 0:
    # Найдём индексы всех непересекающихся участков
    unique_labels = np.unique(labeled_areas[labeled_areas > 0])
    
    # Создаём список цветов
    colors = pc.qualitative.Plotly * len(unique_labels)

    # Отсортируем по размеру
    sizes = []
    for label_num in unique_labels:
        area_mask = labeled_areas == label_num
        size = np.sum(area_mask)
        sizes.append(size)
        
    # Упорядочим участки по размерам
    sorted_indices = np.argsort(sizes)[::-1]
    
    # Отрисовываем ТОЛЬКО самые большие участки
    for i, label_num in enumerate([unique_labels[j] for j in sorted_indices]):
        area_mask = labeled_areas == label_num
        fig.add_trace(go.Contour(
            x=x_coords[0],
            y=y_coords[:, 0],
            z=area_mask.astype(int),
             colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors[i % len(colors)]]],
            contours=dict(start=0.5, end=0.5, size=1),
            showscale=False
        ))

    # Разбиваем плоские участки на квадраты 10x10 метров
    square_size_meters = 60
    square_size_pixels = int(square_size_meters / resolution) # Размер квадрата в пикселях

    for label_num in unique_labels:
         area_mask = labeled_areas == label_num
         
         # Получаем координаты участков
         rows, cols = np.where(area_mask)
         
         if(len(rows) == 0):
            continue;
         
         # Определяем границы области
         min_row, max_row = np.min(rows), np.max(rows)
         min_col, max_col = np.min(cols), np.max(cols)

         for row in range(min_row, max_row, square_size_pixels):
                for col in range(min_col, max_col, square_size_pixels):
                    # Получаем координаты углов квадрата в пикселях
                    square_row_start, square_row_end = row, min(row + square_size_pixels, max_row)
                    square_col_start, square_col_end = col, min(col + square_size_pixels, max_col)

                    # Создаем маску для текущего квадрата
                    square_mask = np.zeros_like(area_mask, dtype=bool)
                    square_mask[square_row_start:square_row_end, square_col_start:square_col_end] = True
                    
                    # Проверяем, что весь квадрат находится внутри плоской области
                    if np.all(area_mask[square_mask]):

                        # Преобразуем в координаты на графике
                        x_start, y_start = x_coords[square_row_start, square_col_start], y_coords[square_row_start, square_col_start]
                        x_end, y_end = x_coords[square_row_end, square_col_end], y_coords[square_row_end, square_col_end]

                        # Рисуем границу квадрата
                        fig.add_trace(go.Scatter(
                            x=[x_start, x_end, x_end, x_start, x_start],
                            y=[y_start, y_start, y_end, y_end, y_start],
                            mode='lines',
                            line=dict(color='black', width = 1),
                            showlegend=False
                        ))
else:
    print("No flat areas found with the given parameters.")
    
print("Нарисовать сетку на плоских участках. Размер ячеек сетки регулируется")
fig.update_layout(
    xaxis_title='X (метры)',
    yaxis_title='Y (метры)'
)

# Запуск интерактивного отображения
fig.show()
