import requests

def get_search_ddg_tool(query):
    """
    DuckDuckGoのAPIを使用して検索を実行し、結果を返す関数。
    
    Args:
        query (str): 検索クエリ
    
    Returns:
        list: 検索結果のリスト（タイトル、リンク、説明）
    """
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,  # 検索クエリを追加
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        results = []
        for topic in data.get("Results", []) + data.get("RelatedTopics", []):
            if "Text" in topic and "FirstURL" in topic:
                results.append({
                    "title": topic["Text"],
                    "link": topic["FirstURL"],
                    "description": topic["Text"]
                })
        return results
    else:
        return [{"error": "Failed to fetch data from DuckDuckGo"}]
