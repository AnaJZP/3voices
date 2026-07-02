"""
institutional_collector.py -- Recolector de discursos institucionales
=====================================================================

Recopila discursos y declaraciones de fuentes institucionales oficiales
usando APIs públicas y scraping de HTML estático:

  • UK Government (GOV.UK)  -- Search + Content JSON APIs
  • European Parliament     -- Open Data API / web scraping
  • European Commission     -- RSS + detail pages
  • UN Secretary General    -- HTML scraping (Drupal)
  • WEF (existente)         -- ya recopilados vía YouTube transcripts

Cada fuente se filtra por keywords de sustentabilidad y se normaliza
a un formato común: {text, date, institution, language, url, n_tokens}.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import re
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *
from constants import normalize_group

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────

RAW_INSTITUTIONAL_DIR = RAW_DIR / "institutional"
RAW_INSTITUTIONAL_DIR.mkdir(parents=True, exist_ok=True)

INSTITUTIONAL_CSV = RAW_INSTITUTIONAL_DIR / "institutional_corpus.csv"
CHECKPOINT_EVERY = 25

# Keywords de sustentabilidad (para filtrar discursos relevantes)
SUSTAINABILITY_KEYWORDS = [
    "sustainability", "sustainable development", "climate change",
    "climate crisis", "global warming", "net zero", "carbon",
    "green deal", "green economy", "circular economy",
    "renewable energy", "energy transition", "paris agreement",
    "biodiversity", "environmental", "SDG", "COP26", "COP27",
    "COP28", "COP29", "COP30", "2030 agenda",
    "clean energy", "fossil fuel", "decarbonization",
    "greenhouse gas", "emission", "deforestation",
]

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests

# User-Agent para requests
HEADERS = {
    "User-Agent": "VOZ_SUS-Research/1.0 (Academic research; sustainability discourse analysis)",
    "Accept": "text/html,application/json,application/xml",
}


def _is_sustainability_related(text: str) -> bool:
    """Check if text contains sustainability-related keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SUSTAINABILITY_KEYWORDS)


def _count_tokens(text: str) -> int:
    """Simple whitespace tokenizer for token count."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def _text_hash(text: str) -> str:
    """Generate a hash for deduplication."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _extract_year(date_str: str) -> Optional[int]:
    """Extract year from various date formats."""
    if not date_str:
        return None
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z", "%d %B %Y", "%B %d, %Y", "%Y"]:
        try:
            return datetime.strptime(date_str[:20], fmt).year
        except (ValueError, TypeError):
            continue
    # Fallback: try to find a 4-digit year
    match = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    if match:
        return int(match.group())
    return None


# ═════════════════════════════════════════════════════════════════════
# 1. UK GOV.UK COLLECTOR
# ═════════════════════════════════════════════════════════════════════

class UKGovCollector:
    """Collect speeches from UK Government via GOV.UK APIs."""

    SEARCH_URL = "https://www.gov.uk/api/search.json"
    CONTENT_URL = "https://www.gov.uk/api/content"

    SEARCH_QUERIES = [
        "climate change", "sustainability", "net zero",
        "green economy", "renewable energy", "COP26",
        "environment", "carbon emissions", "biodiversity",
        "sustainable development", "energy transition",
        "Paris Agreement", "clean growth",
    ]

    def __init__(self, max_per_query: int = 50, date_from: str = "2015-01-01"):
        self.max_per_query = max_per_query
        self.date_from = date_from
        self.seen_slugs = set()
        self.docs = []

    def _search(self, query: str, count: int = 20, start: int = 0) -> dict:
        """Search GOV.UK for speeches."""
        params = {
            "q": query,
            "filter_format": "speech",
            "count": count,
            "start": start,
            "order": "-public_timestamp",
        }
        resp = requests.get(self.SEARCH_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get_content(self, slug: str) -> Optional[dict]:
        """Get full speech content via Content API."""
        url = f"{self.CONTENT_URL}/{slug}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"    [WARN] Error fetching {slug}: {e}")
            return None

    def _extract_text(self, content: dict) -> str:
        """Extract clean text from GOV.UK Content API response."""
        details = content.get("details", {})
        body_html = details.get("body", "")
        if not body_html:
            # Some speeches use 'parts' structure
            parts = details.get("parts", [])
            body_html = " ".join(p.get("body", "") for p in parts)
        if not body_html:
            return ""
        soup = BeautifulSoup(body_html, "lxml")
        return soup.get_text(separator=" ", strip=True)

    def collect(self) -> list[dict]:
        """Run the UK GOV collection pipeline."""
        print("\n" + "=" * 60)
        print("  UK GOV.UK -- Speech Collection")
        print("=" * 60)

        for query in self.SEARCH_QUERIES:
            print(f"\n  [SEARCH] Query: '{query}'")
            collected_this_query = 0

            for start in range(0, self.max_per_query, 20):
                try:
                    data = self._search(query, count=20, start=start)
                except requests.RequestException as e:
                    print(f"    [WARN] Search error: {e}")
                    break

                results = data.get("results", [])
                if not results:
                    break

                for item in results:
                    slug = item.get("link", "").lstrip("/")
                    if not slug or slug in self.seen_slugs:
                        continue

                    title = item.get("title", "")
                    date_str = item.get("public_timestamp", "")

                    # Filter by date
                    year = _extract_year(date_str)
                    if year and year < 2015:
                        continue

                    # Check title for sustainability relevance
                    if not _is_sustainability_related(title + " " + item.get("description", "")):
                        continue

                    self.seen_slugs.add(slug)
                    time.sleep(REQUEST_DELAY)

                    content = self._get_content(slug)
                    if not content:
                        continue

                    text = self._extract_text(content)
                    n_tokens = _count_tokens(text)

                    if n_tokens < MIN_TOKENS:
                        continue

                    self.docs.append({
                        "text": text,
                        "title": title,
                        "date": date_str[:10] if date_str else "",
                        "year": year,
                        "institution": "UK_Government",
                        "language": "en",
                        "url": f"https://www.gov.uk/{slug}",
                        "n_tokens": n_tokens,
                        "source": "institutional",
                    })
                    collected_this_query += 1

                time.sleep(REQUEST_DELAY)

            print(f"    [OK] {collected_this_query} speeches collected")

        print(f"\n  [STATS] Total UK GOV: {len(self.docs)} speeches")
        return self.docs


# ═════════════════════════════════════════════════════════════════════
# 2. EUROPEAN PARLIAMENT COLLECTOR
# ═════════════════════════════════════════════════════════════════════

class EuroparlCollector:
    """Collect plenary debate texts from the European Parliament.

    Uses the Europarl website's debate pages (static HTML)
    and filters for sustainability-related debates.
    """

    BASE_URL = "https://www.europarl.europa.eu"

    SEARCH_QUERIES = [
        "climate change", "European Green Deal", "sustainability",
        "fit for 55", "biodiversity", "renewable energy",
        "circular economy", "carbon border", "net zero",
        "energy transition", "deforestation", "Paris Agreement",
    ]

    def __init__(self, max_results: int = 100, date_from: str = "2015-01-01"):
        self.max_results = max_results
        self.date_from = date_from
        self.seen_urls = set()
        self.docs = []

    def _search_debates(self, query: str) -> list[dict]:
        """Search EP plenary debates via the search endpoint."""
        results = []
        search_url = f"{self.BASE_URL}/plenary/en/debates-video.html"

        # Try the EP search/API approach
        api_url = "https://www.europarl.europa.eu/doceo/document/CRE-9-{date}-{item}_EN.html"

        # Alternative: use the EP website search
        search_api = f"{self.BASE_URL}/en/search/advanced"
        params = {
            "q": query,
            "sort": "DATE_DOCUMENT_DESC",
        }

        try:
            resp = requests.get(
                f"{self.BASE_URL}/doceo/document/CRE-9-2024-01-15_EN.html",
                headers=HEADERS, timeout=30,
            )
            if resp.status_code == 200:
                # Parse for debate links
                pass
        except Exception:
            pass

        return results

    def _scrape_ep_search(self) -> list[dict]:
        """Scrape EP proceedings via Google-indexed pages."""
        print("\n  [SEARCH] Searching European Parliament debates...")

        for query in self.SEARCH_QUERIES:
            # Use GOV.UK-style approach but for EP
            url = f"{self.BASE_URL}/doceo/document/CRE-9-2024-01-15_EN.html"
            # EP debates are organized by date and agenda item
            # We'll use their RSS/Atom feeds when available

            try:
                # Try the EP RSS feed for plenary sessions
                rss_url = f"{self.BASE_URL}/rss/doc/calendrier-session/rss.xml"
                resp = requests.get(rss_url, headers=HEADERS, timeout=30)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, "lxml")
                    items = soup.find_all("item")
                    for item in items[:self.max_results]:
                        link = item.find("link")
                        title = item.find("title")
                        date_tag = item.find("pubdate") or item.find("dc:date")
                        if link and title:
                            yield {
                                "url": link.text.strip() if link.text else "",
                                "title": title.text.strip() if title.text else "",
                                "date": date_tag.text.strip() if date_tag and date_tag.text else "",
                            }
            except Exception as e:
                print(f"    [WARN] EP RSS error: {e}")

    def _fetch_debate_text(self, url: str) -> str:
        """Fetch full text of a debate page."""
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")

            # Try various selectors for EP debate content
            for selector in [
                ".contents", ".doc-format-rendition",
                "article", ".ep_gridcolumn-content",
                "main", "#docBody",
            ]:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(separator=" ", strip=True)
                    if len(text) > 200:
                        return text

            # Fallback: get all paragraphs
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
            return text
        except Exception as e:
            print(f"    [WARN] Error fetching EP debate: {e}")
            return ""

    def collect(self) -> list[dict]:
        """Run the European Parliament collection pipeline."""
        print("\n" + "=" * 60)
        print("  European Parliament -- Debate Collection")
        print("=" * 60)

        # Strategy: fetch known CRE (verbatim report) pages
        # CRE pages follow pattern: /doceo/document/CRE-9-YYYY-MM-DD_EN.html
        # We'll generate URLs for plenary sessions and filter by sustainability keywords

        # EP 9th term: 2019-07 to 2024-07
        # EP 10th term: 2024-07 onwards
        # For 2015-2019 (8th term): CRE-8-YYYY-MM-DD

        collected = 0

        # Fetch recent debates by constructing CRE URLs
        # We'll look for debate transcripts from known sustainability-heavy sessions
        debate_pages = self._get_sustainability_debate_urls()

        for page_info in debate_pages:
            if collected >= self.max_results:
                break

            url = page_info["url"]
            if url in self.seen_urls:
                continue
            self.seen_urls.add(url)

            time.sleep(REQUEST_DELAY)
            text = self._fetch_debate_text(url)

            if not text or not _is_sustainability_related(text[:5000]):
                continue

            n_tokens = _count_tokens(text)
            if n_tokens < MIN_TOKENS:
                continue

            year = _extract_year(page_info.get("date", ""))

            self.docs.append({
                "text": text,
                "title": page_info.get("title", "EP Plenary Debate"),
                "date": page_info.get("date", ""),
                "year": year,
                "institution": "European_Parliament",
                "language": "en",
                "url": url,
                "n_tokens": n_tokens,
                "source": "institutional",
            })
            collected += 1
            if collected % 10 == 0:
                print(f"    [OK] {collected} debates collected...")

        print(f"\n  [STATS] Total EP: {len(self.docs)} debates")
        return self.docs

    def _get_sustainability_debate_urls(self) -> list[dict]:
        """Generate URLs for known sustainability-related EP debates.

        Strategy: EP publishes debate schedules and verbatim reports.
        We search their document system for debates mentioning our keywords.
        """
        urls = []

        # Use EP Oeil legislative observatory search
        # Or EP plenary minutes search
        search_url = "https://www.europarl.europa.eu/plenary/en/texts-adopted.html"

        for query in self.SEARCH_QUERIES[:6]:
            try:
                # Search EP website
                resp = requests.get(
                    f"https://www.europarl.europa.eu/plenary/en/search-plenary.html",
                    params={"query": query},
                    headers=HEADERS,
                    timeout=30,
                )
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, "lxml")
                    links = soup.find_all("a", href=True)
                    for link in links:
                        href = link.get("href", "")
                        if "CRE" in href or "debate" in href.lower():
                            full_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"
                            title_text = link.get_text(strip=True)
                            urls.append({
                                "url": full_url,
                                "title": title_text or f"EP debate: {query}",
                                "date": "",
                            })
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                print(f"    [WARN] EP search error for '{query}': {e}")

        return urls


# ═════════════════════════════════════════════════════════════════════
# 3. EUROPEAN COMMISSION COLLECTOR
# ═════════════════════════════════════════════════════════════════════

class ECCommissionCollector:
    """Collect speeches from the European Commission Press Corner."""

    PRESS_CORNER_URL = "https://ec.europa.eu/commission/presscorner"
    RSS_URL = "https://ec.europa.eu/commission/presscorner/api/rss"

    SEARCH_QUERIES = [
        "climate", "sustainability", "green deal",
        "energy transition", "circular economy", "biodiversity",
        "fit for 55", "carbon", "renewable", "net zero",
    ]

    def __init__(self, max_results: int = 100, date_from: str = "2015-01-01"):
        self.max_results = max_results
        self.date_from = date_from
        self.seen_urls = set()
        self.docs = []

    def _fetch_speech_text(self, url: str) -> str:
        """Fetch full text from EC Press Corner speech page."""
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")

            # EC Press Corner uses Europa Component Library
            for selector in [
                ".ecl-field--name-body",
                ".ecl-page-content",
                ".ecl-container",
                "article",
                'div[role="main"]',
                "main",
            ]:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(separator=" ", strip=True)
                    if len(text) > 200:
                        return text

            # Fallback
            paragraphs = soup.find_all("p")
            return " ".join(p.get_text(strip=True) for p in paragraphs)
        except Exception as e:
            print(f"    [WARN] Error fetching EC speech: {e}")
            return ""

    def _search_speeches(self) -> list[dict]:
        """Search for speeches via the Press Corner listing."""
        results = []

        for query in self.SEARCH_QUERIES:
            # EC Press Corner search
            try:
                search_url = f"{self.PRESS_CORNER_URL}/home/en"
                params = {
                    "keywords": query,
                    "dotyp": "SPEECH",  # Filter by type = Speech
                }
                resp = requests.get(
                    search_url, params=params,
                    headers=HEADERS, timeout=30,
                )
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, "lxml")
                    # Look for speech links
                    links = soup.find_all("a", href=True)
                    for link in links:
                        href = link.get("href", "")
                        if "SPEECH" in href.upper():
                            full_url = href if href.startswith("http") else f"https://ec.europa.eu{href}"
                            if full_url not in self.seen_urls:
                                results.append({
                                    "url": full_url,
                                    "title": link.get_text(strip=True),
                                })
                                self.seen_urls.add(full_url)

                time.sleep(REQUEST_DELAY)
            except Exception as e:
                print(f"    [WARN] EC search error for '{query}': {e}")

        # Also try constructing known SPEECH URLs
        # Pattern: SPEECH_YY_NNNN
        # Try a range of known speech IDs for sustainability commissioners
        print(f"    Found {len(results)} speech URLs via search")
        return results

    def collect(self) -> list[dict]:
        """Run the European Commission collection pipeline."""
        print("\n" + "=" * 60)
        print("  European Commission -- Speech Collection")
        print("=" * 60)

        speech_urls = self._search_speeches()
        collected = 0

        for info in speech_urls:
            if collected >= self.max_results:
                break

            url = info["url"]
            time.sleep(REQUEST_DELAY)
            text = self._fetch_speech_text(url)

            if not text or not _is_sustainability_related(text[:5000]):
                continue

            n_tokens = _count_tokens(text)
            if n_tokens < MIN_TOKENS:
                continue

            # Extract date from URL pattern SPEECH_YY_NNNN
            date_str = ""
            year = None
            match = re.search(r"SPEECH[/_](\d{2})[/_](\d+)", url, re.IGNORECASE)
            if match:
                yy = int(match.group(1))
                year = 2000 + yy if yy < 50 else 1900 + yy
                date_str = f"{year}"

            self.docs.append({
                "text": text,
                "title": info.get("title", "EC Speech"),
                "date": date_str,
                "year": year,
                "institution": "European_Commission",
                "language": "en",
                "url": url,
                "n_tokens": n_tokens,
                "source": "institutional",
            })
            collected += 1

        print(f"\n  [STATS] Total EC: {len(self.docs)} speeches")
        return self.docs


# ═════════════════════════════════════════════════════════════════════
# 4. UN SECRETARY GENERAL COLLECTOR
# ═════════════════════════════════════════════════════════════════════

class UNSGCollector:
    """Collect statements from the UN Secretary General."""

    BASE_URL = "https://www.un.org"
    STATEMENTS_URL = "https://www.un.org/sg/en/latest/sg-statements"

    def __init__(self, max_pages: int = 20, max_results: int = 100):
        self.max_pages = max_pages
        self.max_results = max_results
        self.seen_urls = set()
        self.docs = []

    def _list_statements(self, page: int = 0) -> list[dict]:
        """Get list of statement URLs from a listing page."""
        results = []
        url = f"{self.STATEMENTS_URL}?page={page}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")

            # Drupal views: look for items in the listing
            items = soup.select(".views-row, .view-content article, .node--type-statement")
            if not items:
                # Fallback: look for any links that point to /sg/en/content/
                items = soup.find_all("a", href=re.compile(r"/sg/en/content/"))

            for item in items:
                if isinstance(item, type(soup.new_tag("a"))):
                    link = item
                else:
                    link = item.find("a", href=True)

                if not link or not link.get("href"):
                    continue

                href = link.get("href", "")
                full_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"

                title = link.get_text(strip=True)

                # Look for date
                date_tag = None
                if hasattr(item, "find"):
                    date_tag = item.find("time") or item.find("span", class_="date")
                date_str = date_tag.get("datetime", date_tag.get_text(strip=True)) if date_tag else ""

                results.append({
                    "url": full_url,
                    "title": title,
                    "date": date_str,
                })

        except Exception as e:
            print(f"    [WARN] Error listing page {page}: {e}")

        return results

    def _fetch_statement_text(self, url: str) -> str:
        """Fetch full text of a UN SG statement."""
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")

            for selector in [
                ".field--name-body",
                ".node__content",
                "article",
                'div[role="main"]',
                "main",
            ]:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(separator=" ", strip=True)
                    if len(text) > 200:
                        return text

            paragraphs = soup.find_all("p")
            return " ".join(p.get_text(strip=True) for p in paragraphs)
        except Exception as e:
            print(f"    [WARN] Error fetching statement: {e}")
            return ""

    def collect(self) -> list[dict]:
        """Run the UN SG collection pipeline."""
        print("\n" + "=" * 60)
        print("  UN Secretary General -- Statement Collection")
        print("=" * 60)

        all_items = []
        for page in range(self.max_pages):
            print(f"  [PAGE] Listing page {page}...")
            items = self._list_statements(page)
            if not items:
                print(f"    No more items at page {page}")
                break
            all_items.extend(items)
            time.sleep(REQUEST_DELAY)

        print(f"  Found {len(all_items)} statement URLs")

        collected = 0
        for info in all_items:
            if collected >= self.max_results:
                break

            url = info["url"]
            if url in self.seen_urls:
                continue
            self.seen_urls.add(url)

            time.sleep(REQUEST_DELAY)
            text = self._fetch_statement_text(url)

            if not text or not _is_sustainability_related(text[:5000]):
                continue

            n_tokens = _count_tokens(text)
            if n_tokens < MIN_TOKENS:
                continue

            year = _extract_year(info.get("date", ""))

            self.docs.append({
                "text": text,
                "title": info.get("title", "UN SG Statement"),
                "date": info.get("date", ""),
                "year": year,
                "institution": "UN_Secretary_General",
                "language": "en",
                "url": url,
                "n_tokens": n_tokens,
                "source": "institutional",
            })
            collected += 1
            if collected % 10 == 0:
                print(f"    [OK] {collected} statements collected...")

        print(f"\n  [STATS] Total UN SG: {len(self.docs)} statements")
        return self.docs


# ═════════════════════════════════════════════════════════════════════
# 5. MERGE AND ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════

def load_existing_wef() -> pd.DataFrame:
    """Load existing WEF transcripts from the political corpus."""
    political_csv = RAW_DIR / "political" / "political_corpus.csv"
    if not political_csv.exists():
        print("  [WARN] No existing WEF data found")
        return pd.DataFrame()

    df = pd.read_csv(political_csv, encoding="utf-8-sig")
    df["institution"] = "World_Economic_Forum"
    df["source"] = "institutional"

    # Ensure required columns
    if "n_tokens" not in df.columns:
        df["n_tokens"] = df["text"].apply(_count_tokens)
    if "url" not in df.columns:
        df["url"] = ""
    if "year" not in df.columns:
        df["year"] = df.get("date", pd.Series(dtype=str)).apply(
            lambda x: _extract_year(str(x)) if pd.notna(x) else None
        )

    print(f"  [DIR] Loaded {len(df)} existing WEF transcripts")
    return df


def run_collection(
    max_uk: int = 100,
    max_ep: int = 80,
    max_ec: int = 60,
    max_un: int = 60,
) -> pd.DataFrame:
    """Run all collectors and merge results."""
    print("\n" + "=" * 70)
    print("  INSTITUTIONAL CORPUS COLLECTION")
    print("  Target: ≥150 documents from ≥3 institutions")
    print("=" * 70)

    all_docs = []

    # 1. UK Government (easiest, best APIs)
    try:
        uk = UKGovCollector(max_per_query=max_uk)
        all_docs.extend(uk.collect())
    except Exception as e:
        print(f"\n  [ERR] UK GOV collector failed: {e}")

    # 2. UN Secretary General
    try:
        un = UNSGCollector(max_results=max_un)
        all_docs.extend(un.collect())
    except Exception as e:
        print(f"\n  [ERR] UN SG collector failed: {e}")

    # 3. European Commission
    try:
        ec = ECCommissionCollector(max_results=max_ec)
        all_docs.extend(ec.collect())
    except Exception as e:
        print(f"\n  [ERR] EC collector failed: {e}")

    # 4. European Parliament
    try:
        ep = EuroparlCollector(max_results=max_ep)
        all_docs.extend(ep.collect())
    except Exception as e:
        print(f"\n  [ERR] EP collector failed: {e}")

    # Convert to DataFrame
    df_new = pd.DataFrame(all_docs)

    # 5. Merge with existing WEF data
    df_wef = load_existing_wef()

    if not df_new.empty and not df_wef.empty:
        # Align columns
        common_cols = ["text", "title", "date", "year", "institution",
                       "language", "url", "n_tokens", "source"]
        for col in common_cols:
            if col not in df_new.columns:
                df_new[col] = ""
            if col not in df_wef.columns:
                df_wef[col] = ""
        df = pd.concat([df_new[common_cols], df_wef[common_cols]], ignore_index=True)
    elif not df_new.empty:
        df = df_new
    elif not df_wef.empty:
        df = df_wef
    else:
        print("\n  [ERR] No documents collected!")
        return pd.DataFrame()

    # Deduplicate by text hash
    df["_hash"] = df["text"].apply(_text_hash)
    n_before = len(df)
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"\n  [DEDUP] Removed {n_dupes} duplicates")

    # Save
    df.to_csv(INSTITUTIONAL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  [SAVE] Saved: {INSTITUTIONAL_CSV}")

    # Summary
    print("\n" + "=" * 60)
    print("  COLLECTION SUMMARY")
    print("=" * 60)
    print(f"  Total documents: {len(df)}")
    print(f"  Institutions:    {df['institution'].nunique()}")
    print(f"\n  By institution:")
    for inst, count in df["institution"].value_counts().items():
        years = df[df["institution"] == inst]["year"].dropna()
        yr_range = f"{int(years.min())}–{int(years.max())}" if not years.empty else "N/A"
        print(f"    {inst}: {count} docs ({yr_range})")

    return df


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect institutional speeches")
    parser.add_argument("--max-uk", type=int, default=100, help="Max UK GOV speeches per query")
    parser.add_argument("--max-ep", type=int, default=80, help="Max EP debates")
    parser.add_argument("--max-ec", type=int, default=60, help="Max EC speeches")
    parser.add_argument("--max-un", type=int, default=60, help="Max UN SG statements")
    args = parser.parse_args()

    df = run_collection(
        max_uk=args.max_uk,
        max_ep=args.max_ep,
        max_ec=args.max_ec,
        max_un=args.max_un,
    )

    if not df.empty:
        print(f"\n  [OK] Collection complete: {len(df)} documents from {df['institution'].nunique()} institutions")
    else:
        print("\n  [ERR] Collection yielded no documents")
