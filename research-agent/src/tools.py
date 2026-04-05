import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient

from . import config


class WebSearchTool:
    """Search the web for information relevant to a research question."""

    def __init__(self):
        self.client = TavilyClient(api_key=config.TAVILY_API_KEY)

    def search(self, query: str, max_results: int = config.WEB_SEARCH_MAX_RESULTS) -> list[dict]:
        """Search the web using Tavily API."""
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                include_answer=True,
            )

            results = []
            if "results" in response:
                for result in response["results"]:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "url": result.get("url", ""),
                    })

            return results
        except Exception as e:
            print(f"[ERROR] Web search failed: {e}")
            return []


class PageFetchTool:
    """Fetch and extract text from a web page."""

    def __init__(self, timeout: int = config.WEB_SEARCH_TIMEOUT):
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def fetch_and_extract(self, url: str) -> Optional[str]:
        """Fetch a URL and extract text content."""
        try:
            response = self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "lxml")
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(line for line in lines if line)

            max_chars = 5000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[... truncated ...]"

            return text
        except httpx.RequestError as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to extract text from {url}: {e}")
            return None

    def close(self):
        self.client.close()


def create_tools() -> dict:
    """Create and return all tools."""
    return {
        "search": WebSearchTool(),
        "fetch": PageFetchTool(),
    }
