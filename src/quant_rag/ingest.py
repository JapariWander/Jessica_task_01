"""Public-source acquisition; never follows a failed request with a bypass."""
import csv
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .io import ROOT, digest, now, write_json

USER_AGENT = "QuantResearchCourseRAG/0.1 (public educational research; Python requests)"


def clean_html(raw):
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    author = soup.select_one('meta[name="author"]')
    published = soup.select_one('meta[property="article:published_time"]')
    license_tag = soup.select_one('a[rel~="license"]')
    for tag in soup.select("script,style,nav,footer,header,aside,form,noscript"):
        tag.decompose()
    body = soup.select_one(".post-body") or soup.select_one(".article-content") or soup.select_one("article") or soup.select_one("main")
    if body is None:
        raise ValueError("No article/main container: manual extraction rule needed")
    lines = []
    for el in body.find_all(["h1", "h2", "h3", "p", "pre", "li"]):
        if el.find_parent(["pre", "li"]):
            continue
        text = el.get_text("\n" if el.name == "pre" else " ", strip=True)
        if text:
            prefix = "#" * int(el.name[1]) + " " if el.name.startswith("h") else ""
            lines.append(prefix + text)
    text = "\n\n".join(dict.fromkeys(lines))
    return text, {
        "observed_title": title,
        "observed_author": author.get("content") if author else None,
        "observed_publication_date": published.get("content") if published else None,
        "observed_license_url": license_tag.get("href") if license_tag else None,
    }


def clean_markdown(raw):
    text = raw.decode("utf-8")
    if text.startswith("---\n"):
        text = text.split("---", 2)[2].strip()
    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.I)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.S)
    return text.strip(), {"extraction": "original blog Markdown; images and front matter removed"}


def clean_pdf(raw):
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for n, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        pages.append(f"## PDF page {n}\n\n{text.strip()}")
    return "\n\n".join(pages), {"pdf_pages": len(reader.pages)}


def acquire(manifest=None, output=None, session=None, delay=1.0):
    manifest = Path(manifest or ROOT / "data/sources.csv")
    output = Path(output or ROOT / "data")
    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    results, robots = [], {}
    for row in rows:
        doc_id, url = row["id"], row.get("fetch_url") or row["url"]
        record = {**row, "attempted_at": now()}
        try:
            parsed = urlparse(url)
            if parsed.scheme != "https":
                raise ValueError("Only public HTTPS sources are supported")
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if origin not in robots:
                rr = session.get(origin + "/robots.txt", timeout=30)
                if rr.status_code == 404:
                    robots[origin] = None
                else:
                    rr.raise_for_status()
                    rp = RobotFileParser()
                    rp.parse(rr.text.splitlines())
                    robots[origin] = rp
            if robots[origin] and not robots[origin].can_fetch(USER_AGENT, url):
                raise PermissionError("robots.txt disallows automated acquisition")
            time.sleep(delay)
            response = session.get(url, timeout=60)
            response.raise_for_status()
            raw = response.content
            if row.get("format") == "markdown":
                text, meta = clean_markdown(raw)
                suffix = ".md"
            elif raw.startswith(b"%PDF"):
                text, meta = clean_pdf(raw)
                suffix = ".pdf"
            else:
                text, meta = clean_html(raw)
                suffix = ".html"
            if len(text.split()) < 250:
                raise ValueError("Body too short (<250 words); not accepted as a full document")
            text = re.sub(r"[ \t]+", " ", text).strip()
            body_sha = digest(text.encode())
            record.update(meta)
            record.update(status="ok", fetched_url=response.url, words=len(text.split()),
                          raw_sha256=digest(raw), text_sha256=body_sha)
            (output / "raw").mkdir(parents=True, exist_ok=True)
            (output / "processed").mkdir(parents=True, exist_ok=True)
            (output / "raw" / (doc_id + suffix)).write_bytes(raw)
            document = {**record, "text": text}
            write_json(output / "processed" / (doc_id + ".json"), document)
            (output / "processed" / (doc_id + ".md")).write_text(
                f"# {row['title']}\n\nSource: {url}\n\nLicense: {row['license_note']}\n\n{text}\n",
                encoding="utf-8")
        except (requests.RequestException, ValueError, PermissionError) as exc:
            # Do not write headers/credentials or raw server responses to logs.
            record.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        results.append(record)
        print(f"{doc_id}: {record['status']}", flush=True)
        write_json(output / "acquisition.json", {"created_at": now(), "documents": results})
    return results


def load_documents(directory=None):
    directory = Path(directory or ROOT / "data/processed")
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]
    if not docs:
        raise ValueError("No acquired documents. Run quant-rag ingest or import permitted documents first.")
    ids = [d["id"] for d in docs]
    hashes = [digest(d["text"].encode()) for d in docs]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate document IDs or duplicate bodies")
    return docs
