import os
import time
from pathlib import Path

import requests

BASE_URL = "https://data.sec.gov"

# SEC EDGAR requires a descriptive User-Agent with a real contact email.
# Override via the SEC_EDGAR_USER_AGENT environment variable.
_DEFAULT_AGENT = "finrag-eval contact@example.com"


def _headers() -> dict[str, str]:
    return {"User-Agent": os.environ.get("SEC_EDGAR_USER_AGENT", _DEFAULT_AGENT)}


def get_cik_from_ticker(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=_headers()).json()
    for item in data.values():
        if item["ticker"] == ticker:
            return str(item["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for {ticker}")


def fetch_company_submissions(cik: str) -> dict:
    url = f"{BASE_URL}/submissions/CIK{cik}.json"
    return requests.get(url, headers=_headers()).json()


def filter_filings(
    submissions: dict,
    form_types: tuple[str, ...] = ("10-K", "10-Q"),
    limit: int = 3,
    scan_limit: int = 1000,
) -> list[dict]:
    """Return up to `limit` filings matching `form_types`.

    Scans up to `scan_limit` recent filings to handle prolific filers like JPM
    that file hundreds of prospectus supplements (424B2) daily, which would
    otherwise exhaust a small limit before any 10-K or 10-Q is found.
    """
    filings = submissions["filings"]["recent"]
    results = []
    for i in range(min(scan_limit, len(filings["form"]))):
        if filings["form"][i] in form_types:
            results.append(
                {
                    "form": filings["form"][i],
                    "accession": filings["accessionNumber"][i].replace("-", ""),
                    "primary_doc": filings["primaryDocument"][i],
                    "filing_date": filings["filingDate"][i],
                }
            )
        if len(results) >= limit:
            break
    return results


def download_filing(cik: str, filing: dict, out_dir: Path) -> Path:
    accession = filing["accession"]
    doc = filing["primary_doc"]
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / doc
    r = requests.get(url, headers=_headers())
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def fetch_sec_filings_for_ticker(ticker: str, output_root: str = "data/raw/sec") -> list[str]:
    cik = get_cik_from_ticker(ticker)
    submissions = fetch_company_submissions(cik)
    filings = filter_filings(submissions)
    paths = []
    for filing in filings:
        path = download_filing(cik, filing, Path(output_root) / ticker)
        paths.append(str(path))
        time.sleep(0.2)
    return paths
