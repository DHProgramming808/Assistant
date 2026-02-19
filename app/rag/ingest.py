import httpx
from bs4 import BeautifulSoup
from fastapi import HTTPException

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    for tag in soup(["script", "style", "svg"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

async def fetch_url_text(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) " 
            "Chrome/58.0.3029.110 Safari/537.3"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com" 
    }

    async with httpx.AsyncClient(timeout = 30.0, follow_redirects = True, headers = headers) as client:
        r = await client.get(url)

        print("DEBUG STATUS:", r.status_code)
        print("DEBUG FINAL URL:", str(r.url))
        print("DEBUG SERVER HEADER:", r.headers.get("server"))
        print("DEBUG CONTENT TYPE:", r.headers.get("content-type"))

        r.raise_for_status()
        return extract_text_from_html(r.text)
    
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    i = 0
    while i < len(text):
        end = min(len(text), i + chunk_size)
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = max(0, end - chunk_overlap)

    return chunks

