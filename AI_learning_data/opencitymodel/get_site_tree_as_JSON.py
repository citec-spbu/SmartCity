import os
import re
import requests
import json
from bs4 import BeautifulSoup

output_file = "AI_learning_data/opencitymodel/all_OCM_site_tree.json"
state_links_from_github_path = "AI_learning_data/opencitymodel/all_states_links_from_github.txt"

def correct_url(state_url):
    # if the URL starts with the incorrect double prefix
    if state_url.startswith("http://htmlpreview.github.io/?http://htmlpreview.github.io/?"):
        state_url = state_url[len("http://htmlpreview.github.io/?http://htmlpreview.github.io/?"):]

    # if the URL starts with a single htmlpreview prefix
    elif state_url.startswith("http://htmlpreview.github.io/?"):
        state_url = state_url[len("http://htmlpreview.github.io/?"):]

    # convert GitHub URL to raw.githubusercontent.com format
    if state_url.startswith("https://github.com/"):
        state_url = state_url.replace(
            "https://github.com/",
            "https://raw.githubusercontent.com/"
        ).replace("/blob/", "/")

    return state_url
def extract_cities_data_from_html_soup(soup):
    city_links = {}
    city_headers = soup.find_all("th", {"colspan": "2"})  # find headers with city names

    for header in city_headers:
        # extract city name and FIPS code
        city_info = header.text.strip()
        if "files" in city_info:  # Skip summary rows
            continue
        city_name, _, fips_code = city_info.rpartition(" (")
        fips_code = fips_code.rstrip(")")  # remove the trailing parenthesis

        # find the next tbody element and extract links
        tbody = header.find_next("tbody")
        if not tbody:  # skip if tbody is not found
            continue
        tds = tbody.find_all("td")
        if len(tds) < 2:  # ensure there are at least two columns
            continue
        json_column = tds[1]  # JSON links are in the second column
        json_links = [a["href"] for a in json_column.find_all("a")]

        city_links[city_name] = {
            "fips_code": fips_code,
            "json_links": json_links
        }
    return city_links
def get_cities_data(state_url):
    try:
        state_url = correct_url(state_url)
        print(f"Fetching from corrected URL: {state_url}")

        # fetch and parse the HTML content
        response = requests.get(state_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            cities_data = extract_cities_data_from_html_soup(soup)

            return cities_data
        else:
            print(f"Failed to fetch {state_url} (Status Code: {response.status_code})")
            return {}
    except Exception as e:
        print(f"Error fetching {state_url}: {e}")
        return {}

with open(state_links_from_github_path, 'r') as f:
    state_links_lines = f.readlines()

# regular expression to match state names and their URLs
state_link_pattern = re.compile(r'\[([^\]]+)\]\((http[^\)]+)\)')

# dictionary to store all states and their corresponding cities data
state_cities_data = {}

# loop through each state link
for state_link_line in state_links_lines:
    match = state_link_pattern.search(state_link_line)
    if match:
        state_name = match.group(1)
        state_html_url = f"http://htmlpreview.github.io/?{match.group(2).strip()}"

        print(f"Processing state: {state_name}")
        print(f"Fetching data from: {state_html_url}")

        cities_data = get_cities_data(state_html_url)

        # store the cities data under the state name
        if cities_data:
            state_cities_data[state_name] = cities_data

# save all data to the JSON file
with open(output_file, "w") as f:
    json.dump(state_cities_data, f, indent=4)

print(f"all data saved to {output_file}")
