import json
import geopandas as gpd

with open('AI_learning_data/opencitymodel/all_OCM_site_tree.json', 'r') as f:
    all_OCM_site_tree = json.load(f)

city_fips_from_json = {
    state: {
        city: data['fips_code']
        for city, data in state_data.items()
    }
    for state, state_data in all_OCM_site_tree.items()
}

print(f"total cities in OCM JSON: {sum(len(state_data) for state_data in all_OCM_site_tree.values())}")

city_boundaries = gpd.read_file('AI_learning_data/500_Cities/CityBoundaries.shp')

# extract city names, their PLACEFIPS, and STFIPS
city_fips_from_shapefile = {
    row['NAME'].lower().strip(): (row['PLACEFIPS'], row['STFIPS'])
    for _, row in city_boundaries.iterrows()
}

print(f"total cities in 500 Cities Boundaries dataset: {len(city_fips_from_shapefile)}")

matching_cities = {}
for state, state_data in city_fips_from_json.items():
    for city, fips_code in state_data.items():
        normalized_city = city.lower().strip()
        if normalized_city in city_fips_from_shapefile:
            shapefile_fips, shapefile_state_fips = city_fips_from_shapefile[normalized_city]

            # compare state FIPS codes (first 2 digits of fips_code and STFIPS)
            if fips_code[:2] == str(shapefile_state_fips).zfill(2):  # ensure STFIPS is 2 digits
                matching_cities[normalized_city] = {
                    "fips_code": fips_code,
                    "state_fips_code": str(fips_code[:2])  # state FIPS code (first 2 digits)
                }

print(f"total matching cities: {len(matching_cities)}")
print("matching cities:", matching_cities)

# save matching cities to a JSON file
output_file = 'AI_learning_data/matching_cities_with_fips.json'

with open(output_file, 'w') as f:
    json.dump(matching_cities, f, indent=4)

print(f"matching cities with FIPS codes saved to {output_file}")
