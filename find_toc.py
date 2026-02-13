#!/usr/bin/env python3
"""
Find Table of Contents in an SEC EDGAR 10-K HTML document.
Usage:
  python find_toc.py
  python find_toc.py "https://www.sec.gov/Archives/edgar/data/2012383/000095017025026584/blk-20241231.htm"
"""

import os
import re
import sys
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# SEC requires a descriptive User-Agent (use SEC_EDGAR_EMAIL from .env if set)
_ua = os.environ.get("SEC_EDGAR_EMAIL", "CompanyName contact@example.com")
HEADERS = {
    "User-Agent": _ua,
    "Accept": "text/html,application/xhtml+xml",
}

DEFAULT_URL = "https://www.sec.gov/Archives/edgar/data/2012383/000095017025026584/blk-20241231.htm"


def fetch_html(url: str) -> str:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def find_toc(html: str) -> list[tuple[str, str]]:
    """
    Find Table of Contents entries. Returns list of (label, href or "").
    """
    soup = BeautifulSoup(html, "html.parser")
    toc_entries = []

    # 1) Look for element with id/class containing toc, contents, table
    for attr in ("id", "class"):
        for tag in soup.find_all(True, **{attr: re.compile(r"toc|content|table\s*of\s*content", re.I)}):
            if not tag.get("class") or (attr == "class" and not any(re.search(r"toc|content", c, re.I) for c in tag.get("class", []))):
                if attr == "id" and not re.search(r"toc|content", tag.get("id", ""), re.I):
                    continue
            # Collect links inside this block (Item 1, Item 7, etc.)
            for a in tag.find_all("a", href=True):
                text = a.get_text(strip=True) or ""
                href = a["href"].strip()
                if text and (re.search(r"item\s*\d|part\s*[IV]+", text, re.I) or href.startswith("#")):
                    toc_entries.append((text, href))
            if toc_entries:
                return toc_entries

    # 2) Look for heading "Table of Contents" or "Contents" and take following list/links
    for tag in soup.find_all(string=re.compile(r"table\s*of\s*contents|^contents\s*$", re.I)):
        parent = tag.parent
        if parent is None:
            continue
        # Next sibling or parent's next sibling
        block = parent.find_next_sibling() or parent.parent.find_next_sibling() if parent.parent else None
        if block:
            for a in block.find_all("a", href=True):
                text = a.get_text(strip=True) or ""
                href = a["href"].strip()
                if text:
                    toc_entries.append((text, href))
            if toc_entries:
                return toc_entries

    # 3) Collect all links that look like TOC (anchor to item/part)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True) or ""
        if not text:
            continue
        # Anchor link like #item1, #part1, or text like "Item 1", "Part I"
        if href.startswith("#") and re.search(r"item\s*\d|part\s*[IV]+|\d+\.", text, re.I):
            toc_entries.append((text, href))
        if re.search(r"#item\s*\d|#part", href, re.I):
            toc_entries.append((text, href))

    # 4) Fallback: find any list or div that has multiple "Item N" links
    for container in soup.find_all(["div", "nav", "section", "ul", "ol"]):
        links = container.find_all("a", href=True)
        if len(links) < 3:
            continue
        items = []
        for a in links:
            t = a.get_text(strip=True)
            h = a["href"].strip()
            if re.search(r"item\s*\d|part\s*[IV]|\d+\.", t, re.I) or (h.startswith("#") and t):
                items.append((t, h))
        if len(items) >= 3:
            return items

    return toc_entries


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"Fetching: {url}\n")

    try:
        html = fetch_html(url)
    except Exception as e:
        print(f"Error fetching URL: {e}")
        sys.exit(1)

    toc = find_toc(html)
    if not toc:
        print("No Table of Contents block found. Showing links that look like Item/Part anchors:\n")
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            t = a.get_text(strip=True)
            h = a["href"]
            if h.startswith("#") and (t or re.search(r"item|part", h, re.I)):
                toc.append((t or h, h))
        toc = toc[:80]  # limit

    print("Table of Contents (or relevant links):")
    print("-" * 60)
    for label, href in toc:
        print(f"  {label or '(no text)'}\t{href}")
    print("-" * 60)
    print(f"Total: {len(toc)} entries.")


if __name__ == "__main__":
    main()
