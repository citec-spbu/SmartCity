import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, zoom

def create_wide_flat_hilly_dem(output_file, width, height, cell_size, base_elevation, max_elevation_change, sigma, block_size, noise_reduction_factor):
    # Создаем базовый массив
    data = np.full((height, width), base_elevation, dtype=np.float32)

    # Вычисляем размеры разреженного массива
    sparse_height = height // block_size
    sparse_width = width // block_size

    # Создаем разреженный шум
    sparse_noise = np.random.normal(0, max_elevation_change / 3, (sparse_height, sparse_width)).astype(np.float32)
     
    # Уменьшаем силу шума
    sparse_noise = sparse_noise/noise_reduction_factor

    # Увеличиваем размер разреженного шума
    noise = zoom(sparse_noise, (block_size, block_size), order=1)

    # Сглаживаем шум
    smoothed_noise = gaussian_filter(noise, sigma=sigma)

    # Добавляем шум к базовому массиву
    data += smoothed_noise

    transform = from_origin(0, 0, cell_size, cell_size)
    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': transform,
        'nodata': -9999
    }

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data, 1)

if __name__ == "__main__":
    output_dem = "deterministic_algorithms/hilly_dem.dem"
    width = 500
    height = 500
    cell_size = 1.0
    base_elevation = 10.0
    max_elevation_change = 10  # Уменьшаем максимальное изменение высоты
    sigma = 30  # Увеличиваем sigma для более широких холмов
    block_size = 50  # Увеличиваем размер блоков для меньшего количества холмов
    noise_reduction_factor = 0.1 # Уменьшаем силу шума

    create_wide_flat_hilly_dem(output_dem, width, height, cell_size, base_elevation, max_elevation_change, sigma, block_size, noise_reduction_factor)
    print(f"Файл .DEM с широкими и низкими холмами создан: {output_dem}")
