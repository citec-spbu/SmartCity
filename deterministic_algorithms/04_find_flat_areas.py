import numpy as np
import plotly.graph_objects as go
import rasterio
from rasterio.warp import transform_bounds
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import disk
import plotly.colors as pc

def read_dem_data(file_path):
    """Читает DEM данные из GeoTIFF файла."""
    with rasterio.open(file_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    return dem_data, transform, crs, bounds

def create_coordinate_grid(bounds, crs, width, height):
    """Создает координатную сетку в метрических координатах."""
    minx, miny, maxx, maxy = transform_bounds(crs, "+init=epsg:4326", *bounds)
    x_coords = np.linspace(minx, maxx, width)
    y_coords = np.linspace(miny, maxy, height)
    return np.meshgrid(x_coords, y_coords)

def find_flat_areas(dem, min_height=0, max_height_diff=1, min_radius=10, resolution=1):
    """Находит плоские участки на DEM."""
    is_flat = np.zeros_like(dem, dtype=bool)
    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            if dem[i, j] >= min_height:
                window = dem[max(0, i - 1):min(dem.shape[0], i + 2),
                            max(0, j - 1):min(dem.shape[1], j + 2)]
                if np.max(window) - np.min(window) <= max_height_diff:
                    is_flat[i, j] = True

    labeled_flat_areas, num_labels = label(is_flat)

    min_pixel_radius = int(min_radius / resolution)
    filtered_flat_areas = np.zeros_like(labeled_flat_areas)
    labels_to_keep = []
    for label_num in range(1, num_labels + 1):
        area_mask = labeled_flat_areas == label_num
        size = np.sum(area_mask)
        if size >= np.pi * min_pixel_radius**2:
            is_overlapping = False
            for prev_label in labels_to_keep:
                prev_mask = labeled_flat_areas == prev_label
                overlap = np.sum(area_mask & prev_mask)
                if overlap > 0:
                    is_overlapping = True
                    break
            if not is_overlapping:
                labels_to_keep.append(label_num)
                filtered_flat_areas += area_mask * label_num
    return filtered_flat_areas

def create_dem_contour(x_coords, y_coords, dem_data):
    """Создает контурный график для DEM."""
    return go.Contour(
        x=x_coords[0],
        y=y_coords[:, 0],
        z=dem_data,
        colorscale='Viridis',
        contours=dict(start=np.min(dem_data), end=np.max(dem_data), size=25)
    )

def create_flat_area_contours(x_coords, y_coords, labeled_areas):
    """Создает контуры для плоских участков."""
    contours = []
    if np.max(labeled_areas) > 0:
        unique_labels = np.unique(labeled_areas[labeled_areas > 0])
        colors = pc.qualitative.Plotly * len(unique_labels)
        sizes = []
        for label_num in unique_labels:
            area_mask = labeled_areas == label_num
            sizes.append(np.sum(area_mask))
        sorted_indices = np.argsort(sizes)[::-1]
        for i, label_num in enumerate([unique_labels[j] for j in sorted_indices]):
            area_mask = labeled_areas == label_num
            contours.append(go.Contour(
                x=x_coords[0],
                y=y_coords[:, 0],
                z=area_mask.astype(int),
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors[i % len(colors)]]],
                contours=dict(start=0.5, end=0.5, size=1),
                showscale=False
            ))
    return contours

def create_squares(x_coords, y_coords, labeled_areas, resolution, square_sizes_meters):
    """Создает и возвращает список прямоугольников для отрисовки."""
    squares = []
    drawn_squares = set()

    if np.max(labeled_areas) > 0:
      
      unique_labels = np.unique(labeled_areas[labeled_areas > 0])

      for square_size_meters in square_sizes_meters:
        square_size_pixels = int(square_size_meters / resolution)

        for label_num in unique_labels:
            area_mask = labeled_areas == label_num

            rows, cols = np.where(area_mask)
            if(len(rows) == 0):
              continue

            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            
            for row in range(min_row, max_row, square_size_pixels):
                for col in range(min_col, max_col, square_size_pixels):
                  
                    square_row_start, square_row_end = row, min(row + square_size_pixels, max_row)
                    square_col_start, square_col_end = col, min(col + square_size_pixels, max_col)
                    square_mask = np.zeros_like(area_mask, dtype=bool)
                    square_mask[square_row_start:square_row_end, square_col_start:square_col_end] = True

                    if np.all(area_mask[square_mask]):

                        x_start, y_start = x_coords[square_row_start, square_col_start], y_coords[square_row_start, square_col_start]
                        x_end, y_end = x_coords[square_row_end, square_col_end], y_coords[square_row_end, square_col_end]
                        square_hash = (x_start, y_start, x_end, y_end)

                        is_overlapping = False
                        for drawn_square_hash in drawn_squares:
                             if not (x_end < drawn_square_hash[0] or x_start > drawn_square_hash[2] or y_end < drawn_square_hash[1] or y_start > drawn_square_hash[3]):
                                is_overlapping = True
                                break;
                        
                        if not is_overlapping:
                            squares.append(go.Scatter(
                                x=[x_start, x_end, x_end, x_start, x_start],
                                y=[y_start, y_start, y_end, y_end, y_start],
                                mode='lines',
                                line=dict(color='black', width = 1 if square_size_meters==10 else 2),
                                showlegend=False
                            ))
                            drawn_squares.add(square_hash)
    return squares

if __name__ == '__main__':
    # Параметры
    file_path = "deterministic_algorithms/hilly_dem.dem"
    min_height = 1
    max_height_diff = 1
    min_radius = 10
    square_sizes_meters = [100, 60, 30]

    # Чтение данных
    dem_data, transform, crs, bounds = read_dem_data(file_path)
    height, width = dem_data.shape
    x_coords, y_coords = create_coordinate_grid(bounds, crs, width, height)
    resolution_x = (np.max(x_coords)-np.min(x_coords))/width
    resolution_y = (np.max(y_coords)-np.min(y_coords))/height
    resolution = (resolution_x+resolution_y)/2

    # Обработка данных
    labeled_areas = find_flat_areas(dem_data, min_height, max_height_diff, min_radius, resolution)
    
    # Создание графика
    fig = go.Figure()
    fig.add_trace(create_dem_contour(x_coords, y_coords, dem_data))
    
    # Добавляем контуры плоских участков
    for trace in create_flat_area_contours(x_coords, y_coords, labeled_areas):
        fig.add_trace(trace)
        
    # Добавляем квадраты
    for trace in create_squares(x_coords, y_coords, labeled_areas, resolution, square_sizes_meters):
        fig.add_trace(trace)

    print("Выделили крупные ровные области.\nВыделили максимальне кол-во участков разного размера")
    fig.update_layout(
        xaxis_title='X (метры)',
        yaxis_title='Y (метры)'
    )

    fig.show()
