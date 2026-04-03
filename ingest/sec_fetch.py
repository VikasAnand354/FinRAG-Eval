import requests
from pathlib import Path
import time

BASE_URL = "https://data.sec.gov"
HEADERS = {"User-Agent": "finrag-eval your_email@example.com"}

def get_cik_from_ticker(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url).json()
    for item in data.values():
        if item["ticker"] == ticker:
            return str(item["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for {ticker}")

def fetch_company_submissions(cik: str):
    url = f"{BASE_URL}/submissions/CIK{cik}.json"
    return requests.get(url, headers=HEADERS).json()

def filter_filings(submissions, form_types=("10-K","10-Q"), limit=3):
    filings = submissions["filings"]["recent"]
    results = []
    for i in range(len(filings["form"])):
        if filings["form"][i] in form_types:
            results.append({
                "form": filings["form"][i],
                "accession": filings["accessionNumber"][i].replace("-", ""),
                "primary_doc": filings["primaryDocument"][i],
                "filing_date": filings["filingDate"][i]
            })
        if len(results) >= limit:
            break
    return results

def download_filing(cik: str, filing, out_dir: Path):
    accession = filing["accession"]
    doc = filing["primary_doc"]
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / doc
    r = requests.get(url, headers=HEADERS)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def fetch_sec_filings_for_ticker(ticker: str, output_root="data/raw/sec"):
    cik = get_cik_from_ticker(ticker)
    submissions = fetch_company_submissions(cik)
    filings = filter_filings(submissions)
    paths = []
    for filing in filings:
        path = download_filing(cik, filing, Path(output_root) / ticker)
        paths.append(str(path))
        time.sleep(0.2)
    return paths
