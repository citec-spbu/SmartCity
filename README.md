# SmartCity

## Обзор

Система генерации застройки.

---

## Шаги по генерации данных для обучения нейросети

### 1. Скачать необходимые данные

1. **запустить файл:**
    ```bash
    python3 AI_learning_data/download_JSONs_of_corresponding_cities.py
    ```
2. **скачать архив 500 Cities: City Boundaries:**
    - [500 Cities: City Boundaries](https://data.cdc.gov/500-Cities-Places/500-Cities-City-Boundaries/n44h-hy2j/about_data)
    - поместить его содержимое в папку:
        ```
        AI_learning/500_Cities
        ```

### 2. Отфильтровать данные

Удалить здания, находящиеся за чертами городов:

```bash
python3 AI_learning_data/filter_buildings_data.py
```

### 3. Соединить файлы

Скомбинировать несколько `.json`-файлов, относящихся к одному городу, в один большой файлик:

```bash
python3 AI_learning_data/combine_filtered_buildings_data.py
```

### 4. Разбить город на полигоны

Запустить алгоритм разбиения города на полигоны разных размеров с сохранением получившихся результатов:

```bash
python3 AI_learning_data/split_city_into_polygons.py
```

### 5. Запустить финальную чистку результатов

Нужно убрать полигоны без зданий, так как они бесполезны для обучения нейросети:

```bash
python3 AI_learning_data/remove_empty_polygons.py
```

---
