# tools/fetch_page.py

import requests

def get_fetch_page_tool(url):
    """指定した URL の HTML を取得"""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return f"Failed to fetch page: {response.status_code}"
