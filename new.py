import requests
from bs4 import BeautifulSoup


def print_grid_from_google_doc(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find a table in the Google Doc.")

    rows = table.find_all("tr")

    grid = {}
    max_x = 0
    max_y = 0
    
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])

        if len(cells) < 3:
            continue

        x = int(cells[0].get_text(strip=True))
        char = cells[1].get_text()
        y = int(cells[2].get_text(strip=True))

        grid[(x, y)] = char

        max_x = max(max_x, x)
        max_y = max(max_y, y)

    for y in range(max_y, -1, -1):
        row = ""

        for x in range(max_x + 1):
            row += grid.get((x, y), " ")

        print(row)

print_grid_from_google_doc("https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub")