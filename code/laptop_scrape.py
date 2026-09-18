

import os
import re
import sys
import time
import json
import random
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_URL = "https://www.flipkart.com/search"

# Search sub-queries to overcome Flipkart's 40-page (~960 item) per-query pagination limit
SEARCH_QUERIES = [
    "laptop",
    "gaming laptop",
    "thin and light laptop",
    "hp laptop",
    "dell laptop",
    "lenovo laptop",
    "asus laptop",
    "acer laptop",
    "apple macbook",
    "i5 laptop",
    "i7 laptop",
    "ryzen laptop",
    "touchscreen laptop",
    "oled laptop",
]

# Standard desktop browser User-Agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# Canonical brand name mappings (preserving uppercase acronyms)
BRAND_CANONICAL = {
    "HP": "HP",
    "DELL": "Dell",
    "LENOVO": "Lenovo",
    "ASUS": "ASUS",
    "ACER": "Acer",
    "APPLE": "Apple",
    "MACBOOK": "Apple",
    "MSI": "MSI",
    "SAMSUNG": "Samsung",
    "LG": "LG",
    "GIGABYTE": "Gigabyte",
    "INFINIX": "Infinix",
    "XIAOMI": "Xiaomi",
    "REDMIBOOK": "Xiaomi",
    "REALME": "Realme",
    "PRIMEBOOK": "Primebook",
    "MICROSOFT": "Microsoft",
    "SURFACE": "Microsoft",
    "CHUWI": "Chuwi",
    "HONOR": "Honor",
    "ZEBRONICS": "Zebronics",
    "ULTIMUS": "Ultimus",
    "VAIO": "VAIO",
    "WINGS": "Wings",
}


def get_headers() -> Dict[str, str]:
    """Generate realistic browser headers with rotated User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }


def create_session() -> requests.Session:
    """Initialize a requests Session with HTTP Keep-Alive."""
    session = requests.Session()
    session.headers.update(get_headers())
    return session


def get_page(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    session: Optional[requests.Session] = None,
    retries: int = 3,
    min_delay: float = 1.5,
    max_delay: float = 3.5,
) -> Optional[BeautifulSoup]:
    """
    Fetch HTML content with session reuse, random jitter delays, and retry logic.
    """
    client = session if session is not None else requests.Session()

    for attempt in range(1, retries + 1):
        try:
            # Respectful random jitter delay between requests
            time.sleep(random.uniform(min_delay, max_delay))

            # Rotate User-Agent per request to prevent fingerprint tracking
            headers = get_headers()
            response = client.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 200:
                if "recaptcha" in response.text.lower() and len(response.text) < 5000:
                    logging.warning("Flipkart served a reCAPTCHA challenge. Backing off...")
                    time.sleep(random.uniform(10.0, 15.0))
                    continue
                return BeautifulSoup(response.text, "html.parser")

            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", random.randint(10, 20)))
                logging.warning(f"Rate limited (HTTP 429). Sleeping for {retry_after}s...")
                time.sleep(retry_after)

            elif response.status_code in (403, 503):
                logging.warning(f"Access challenge (HTTP {response.status_code}) on attempt {attempt}. Backing off...")
                time.sleep(random.uniform(5.0, 10.0))

            else:
                logging.warning(f"HTTP {response.status_code} received on attempt {attempt}")

        except requests.RequestException as e:
            logging.error(f"Network error on attempt {attempt} for URL '{url}': {e}")
            time.sleep(random.uniform(2.0, 4.0))

    return None


def parse_listing(card: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """
    Extract product details and hardware specifications from a single search listing card.
    Supports both Flipkart's modern and legacy CSS class selectors.
    """
    data: Dict[str, Any] = {}

    # 1. Product Name & URL Anchor
    # Modern title classes: RG5Slk, KzI22o, wjcEIp | Legacy: rR01T, _4rR01T, D4L23
    title_elem = card.find(
        ["div", "a"],
        class_=re.compile(r"RG5Slk|KzI22o|wjcEIp|rR01T|_4rR01T|D4L23"),
    )

    product_anchor = card.find("a", href=re.compile(r"/p/itm"))
    if not title_elem and product_anchor and product_anchor.get_text(strip=True):
        title_elem = product_anchor

    if not title_elem or not title_elem.get_text(strip=True):
        return None

    raw_title = title_elem.get_text(strip=True)
    data["Product Name"] = raw_title

    # Relative product page URL for optional deep scraping
    data["Product URL"] = product_anchor["href"] if product_anchor and "href" in product_anchor.attrs else None

    # 2. Brand Extraction & Normalization
    first_word = raw_title.split()[0].upper()
    data["Brand"] = BRAND_CANONICAL.get(first_word, first_word.capitalize())

    # 3. Price (Selling Price)
    # Modern: hZ3P6w, DeU9vF, Nx9bqj | Legacy: _30jeq3, _1_WHN1
    price_elem = card.find("div", class_=re.compile(r"hZ3P6w|DeU9vF|Nx9bqj|_30jeq3|_1_WHN1"))
    data["Price"] = price_elem.get_text(strip=True) if price_elem else None

    # 4. MRP (Maximum Retail Price)
    # Modern: kRYCnD, gxR4EY, yRaATh | Legacy: _3I9_wc|_27R201
    mrp_elem = card.find("div", class_=re.compile(r"kRYCnD|gxR4EY|yRaATh|_3I9_wc|_27R201"))
    data["MRP"] = mrp_elem.get_text(strip=True) if mrp_elem else None

    # 5. Discount
    # Modern: HZ0E6r or containing '% off' | Legacy: Uk9230, _3Ay63b, _31qS2D
    discount_elem = card.find(
        ["span", "div"],
        class_=re.compile(r"Uk9230|_3Ay63b|_31qS2D"),
    )
    if not discount_elem:
        discount_elem = card.find(
            lambda tag: tag.name in ["span", "div"] and "% off" in tag.get_text().lower()
        )
    data["Discount"] = discount_elem.get_text(strip=True) if discount_elem else None

    # 6. Rating (out of 5.0)
    # Modern: MKiFS6, CjyrHS | Legacy: _3LWZlK, X12d
    rating_elem = card.find("div", class_=re.compile(r"MKiFS6|CjyrHS|_3LWZlK|X12d"))
    data["Rating"] = rating_elem.get_text(strip=True) if rating_elem else None

    # 7. Review Count & Rating Count
    # Modern: PvbNMB or span containing 'Ratings' | Legacy: _2_R_2P, _13v1D, _2rcB2p
    reviews_elem = card.find(
        ["span", "div"],
        class_=re.compile(r"PvbNMB|_2_R_2P|_13v1D|_2rcB2p"),
    )
    if not reviews_elem:
        reviews_elem = card.find(
            lambda tag: tag.name in ["span", "div"]
            and ("reviews" in tag.get_text().lower() or "ratings" in tag.get_text().lower())
        )
    data["Review Count"] = reviews_elem.get_text(strip=True) if reviews_elem else None

    # 8. Bullet Specifications (Processor, RAM, Storage, OS, Display, Graphics)
    ul_elem = card.find("ul")
    if ul_elem:
        specs = [li.get_text(strip=True) for li in ul_elem.find_all("li") if li.get_text(strip=True)]
    else:
        li_elems = card.find_all("li")
        if li_elems:
            specs = [li.get_text(strip=True) for li in li_elems if li.get_text(strip=True)]
        else:
            specs = []

    specs_text = " | ".join(specs)
    title_lower = raw_title.lower()

    # Processor
    data["Processor"] = next(
        (
            s for s in specs
            if any(p in s.lower() for p in [
                "intel", "amd", "ryzen", "core", "apple", "m1", "m2", "m3", "m4",
                "snapdragon", "celeron", "pentium", "mediatek", "athlon"
            ]) and "graphics" not in s.lower()
        ),
        None
    )

    # RAM (supports standard DDR/LPDDR RAM & Apple Unified Memory)
    data["RAM"] = next(
        (
            s for s in specs
            if any(r_kw in s.lower() for r_kw in ["ram", "unified memory", "lpddr", "ddr4", "ddr5"])
        ),
        None
    )

    # Storage
    data["Storage_Raw"] = next(
        (
            s for s in specs
            if any(st in s.lower() for st in ["ssd", "hdd", "emmc", "ufs"])
        ),
        None
    )

    # Operating System
    data["Operating System"] = next(
        (
            s for s in specs
            if any(os_kw in s.lower() for os_kw in [
                "windows", "chrome", "mac", "dos", "linux", "ubuntu"
            ])
        ),
        None
    )

    # Display Size
    data["Display Size"] = next(
        (
            s for s in specs
            if any(d_kw in s.lower() for d_kw in ["display", "inch", "cm", "screen", "oled"])
        ),
        None
    )

    # Graphics & GPU
    graphics_bullet = next(
        (
            s for s in specs
            if any(g_kw in s.lower() for g_kw in [
                "graphics", "nvidia", "rtx", "gtx", "geforce", "radeon", "intel iris", "intel arc"
            ])
        ),
        None
    )
    data["Graphics"] = graphics_bullet
    data["GPU"] = graphics_bullet

    # 9. Extended Categorical Fields
    # Laptop Type
    if any(k in title_lower for k in ["2 in 1", "2-in-1", "x360", "convertible", "yoga", "flip"]):
        data["Laptop Type"] = "2-in-1"
    elif any(k in title_lower for k in ["gaming", "rog", "tuf", "nitro", "loq", "predator", "legion", "victus", "omen", "katana"]):
        data["Laptop Type"] = "Gaming"
    elif any(k in title_lower for k in ["thin", "slim", "air", "zenbook", "swift", "gram", "yoga slim", "vivobook s"]):
        data["Laptop Type"] = "Thin & Light"
    else:
        data["Laptop Type"] = "Standard Notebook"

    # Availability
    out_of_stock = card.find(
        string=re.compile(r"out of stock|currently unavailable", re.IGNORECASE)
    )
    data["Availability"] = "Out of Stock" if out_of_stock else "In Stock"

    # Touchscreen
    is_touch = (
        "touchscreen" in specs_text.lower()
        or "touch" in title_lower
        or data["Laptop Type"] == "2-in-1"
    )
    data["Touchscreen"] = "Yes" if is_touch else "No"

    # Laptop Use
    if data["Laptop Type"] == "Gaming" or "gaming" in specs_text.lower():
        data["Laptop Use"] = "Gaming"
    elif any(k in title_lower or k in specs_text.lower() for k in ["creator", "studio", "pro art", "oled"]):
        data["Laptop Use"] = "Content Creation"
    elif any(k in title_lower or k in specs_text.lower() for k in ["thinkpad", "latitude", "expertbook", "probook", "business"]):
        data["Laptop Use"] = "Business"
    else:
        data["Laptop Use"] = "Student / Everyday"

    return data


def parse_product_page(product_url: str, session: requests.Session) -> Dict[str, Any]:
    """
    Fetch and parse deep specifications from an individual Flipkart product page.
    Extracts structured data from embedded window.__INITIAL_STATE__ JSON, avoiding
    the necessity of heavyweight Selenium automation.
    """
    full_url = product_url if product_url.startswith("http") else f"https://www.flipkart.com{product_url}"
    soup = get_page(full_url, session=session, min_delay=1.0, max_delay=2.0)
    if not soup:
        return {}

    deep_specs: Dict[str, Any] = {}

    # Extract embedded JSON state
    html_text = str(soup)
    match = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});</script>", html_text)
    if match:
        try:
            state = json.loads(match.group(1))
            widgets = state.get("multiWidgetState", {}).get("widgetsData", {})

            # Traverse all slot widgets for specification tables
            def find_specs_in_obj(obj: Any):
                if isinstance(obj, dict):
                    if "label_0" in obj and "label_1" in obj:
                        label = obj.get("label_0", {}).get("value", {}).get("text")
                        values = obj.get("label_1", {}).get("value", {}).get("text", [])
                        if label and values:
                            val_str = " ".join(values) if isinstance(values, list) else str(values)
                            deep_specs[label.strip()] = val_str.strip()
                    for v in obj.values():
                        find_specs_in_obj(v)
                elif isinstance(obj, list):
                    for item in obj:
                        find_specs_in_obj(item)

            find_specs_in_obj(widgets)
        except Exception as e:
            logging.debug(f"JSON state parsing error on product page: {e}")

    return deep_specs


def scrape_flipkart(
    target_count: int = 5000,
    deep_scrape: bool = False,
    checkpoint_file: str = "checkpoint_laptops.csv",
) -> pd.DataFrame:
    """
    Iterate through paginated Flipkart search result URLs and extract laptop listings.
    Optionally visits individual product pages for deep specification extraction.
    Periodically checkpoints scraped records to disk.
    """
    session = create_session()
    raw_records: List[Dict[str, Any]] = []

    logging.info(f"Starting Flipkart laptop scraping pipeline. Target: ~{target_count} items.")

    for query_idx, query in enumerate(SEARCH_QUERIES, 1):
        if len(raw_records) >= target_count:
            logging.info(f"Target count of {target_count} reached. Halting discovery.")
            break

        logging.info(f"[{query_idx}/{len(SEARCH_QUERIES)}] Query scope: '{query}'")

        # Flipkart search pagination is capped at page 40 (40 pages * 24 items = 960 items max per query)
        for page in range(1, 41):
            if len(raw_records) >= target_count:
                break

            params = {"q": query, "page": page}
            soup = get_page(BASE_URL, params=params, session=session)

            if not soup:
                logging.warning(f"Could not retrieve page {page} for '{query}'. Skipping.")
                continue

            # Identify laptop product card containers (prioritizing leaf card container to prevent duplicate nested cards)
            cards = (
                soup.find_all("div", class_="nZIRY7")
                or soup.find_all("div", class_=re.compile(r"_75ndR|cPHRSc|_1sdMkc|_2kHMtA"))
                or soup.find_all("div", class_=re.compile(r"col-12-12"))
            )

            # Filter cards containing actual product listings
            valid_cards = [
                c for c in cards
                if c.find(["div", "a"], class_=re.compile(r"RG5Slk|KzI22o|wjcEIp|rR01T|_4rR01T|D4L23"))
            ]

            if not valid_cards:
                logging.info(f"No further items found for '{query}' at page {page}. Moving to next search query.")
                break

            page_extracted = 0
            for card in valid_cards:
                record = parse_listing(card)
                if record and record.get("Product Name"):
                    # Deep scrape product page if requested and product URL exists
                    if deep_scrape and record.get("Product URL"):
                        try:
                            deep_data = parse_product_page(record["Product URL"], session=session)
                            if deep_data.get("Graphic Processor"):
                                record["GPU"] = deep_data["Graphic Processor"]
                            if deep_data.get("RAM"):
                                record["RAM"] = deep_data["RAM"]
                            if deep_data.get("SSD Capacity") or deep_data.get("HDD Capacity"):
                                record["Storage_Raw"] = f"{deep_data.get('SSD Capacity', '')} {deep_data.get('HDD Capacity', '')}".strip()
                            if deep_data.get("Screen Size"):
                                record["Display Size"] = deep_data["Screen Size"]
                        except Exception as e:
                            logging.debug(f"Deep scrape error for {record.get('Product Name')}: {e}")

                    raw_records.append(record)
                    page_extracted += 1

            logging.info(
                f"Query: '{query}' | Page {page} | Extracted: {page_extracted} | Cumulative: {len(raw_records)}"
            )

            # Checkpoint every 200 items to safeguard progress
            if len(raw_records) % 200 < page_extracted and checkpoint_file:
                try:
                    pd.DataFrame(raw_records).to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
                    logging.info(f"Progress checkpoint saved ({len(raw_records)} records) to '{checkpoint_file}'")
                except Exception as e:
                    logging.warning(f"Could not save checkpoint: {e}")

    return pd.DataFrame(raw_records)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, parse, standardize, and format extracted laptop data for Power BI.
    Enforces strict typing across all 19 target columns and ensures zero mixed types.
    """
    if df.empty:
        logging.warning("Input DataFrame is empty. Returning empty cleaned DataFrame.")
        return pd.DataFrame()

    logging.info("Starting comprehensive data cleaning pipeline for Power BI...")
    cleaned_df = pd.DataFrame(index=df.index)

    # 1. Product Name
    cleaned_df["Product Name"] = (
        df["Product Name"]
        .astype(str)
        .str.replace(r"[\t\n\r]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 2. Brand Standardisation
    def clean_brand(val: Any) -> str:
        if pd.isna(val) or val is None:
            return "Not Specified"
        b = str(val).strip()
        b_upper = b.upper()
        if b_upper in BRAND_CANONICAL:
            return BRAND_CANONICAL[b_upper]
        return b.capitalize()

    cleaned_df["Brand"] = df["Brand"].apply(clean_brand)

    # 3. Laptop Type
    valid_types = ["Gaming", "Thin & Light", "2-in-1", "Standard Notebook"]
    cleaned_df["Laptop Type"] = (
        df["Laptop Type"]
        .apply(lambda x: x if x in valid_types else "Standard Notebook")
        .fillna("Standard Notebook")
    )

    # 4. Numeric Currency Extraction (Price & MRP)
    def clean_currency(val: Any) -> float:
        if pd.isna(val) or val is None:
            return np.nan
        val_str = re.sub(r"[^\d.]", "", str(val))
        try:
            return float(val_str) if val_str else np.nan
        except ValueError:
            return np.nan

    cleaned_df["Price"] = df["Price"].apply(clean_currency)
    cleaned_df["MRP"] = df["MRP"].apply(clean_currency)

    # Ensure MRP is at least equal to Price if missing
    cleaned_df["MRP"] = cleaned_df["MRP"].fillna(cleaned_df["Price"])

    # 5. Discount Percentage (Numeric only)
    def extract_discount(row: pd.Series) -> float:
        raw_disc = row.get("Discount")
        if pd.notna(raw_disc):
            match = re.search(r"(\d+\.?\d*)", str(raw_disc))
            if match:
                return float(match.group(1))
        # Recalculate if Price and MRP are both valid floats
        price = row.get("Price")
        mrp = row.get("MRP")
        if pd.notna(mrp) and pd.notna(price) and mrp > 0 and mrp >= price:
            return round(((mrp - price) / mrp) * 100, 2)
        return np.nan

    # Apply row-wise over cleaned_df so numeric floats are guaranteed
    cleaned_df["Discount"] = cleaned_df.apply(extract_discount, axis=1)

    # 6. Rating (Numeric out of 5.0)
    cleaned_df["Rating"] = pd.to_numeric(
        df["Rating"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )

    # 7. Review Count (Numeric integer count)
    def extract_reviews(val: Any) -> float:
        if pd.isna(val) or val is None:
            return np.nan
        val_str = str(val)
        # Match 'X Reviews'
        match = re.search(r"([\d,]+)\s*Reviews", val_str, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
        # Fallback to second number in 'X Ratings & Y Reviews'
        nums = re.findall(r"[\d,]+", val_str)
        if len(nums) >= 2:
            return float(nums[1].replace(",", ""))
        elif len(nums) == 1 and "review" in val_str.lower():
            return float(nums[0].replace(",", ""))
        return np.nan

    cleaned_df["Review Count"] = df["Review Count"].apply(extract_reviews)

    # 8. Processor Standardisation
    def clean_processor(val: Any) -> str:
        if pd.isna(val) or val is None or str(val).strip().lower() in ["none", "nan", ""]:
            return "Not Specified"
        p = str(val).strip()
        p = re.sub(r"Processor$", "", p, flags=re.IGNORECASE).strip()
        return p if p else "Not Specified"

    cleaned_df["Processor"] = df["Processor"].apply(clean_processor)

    # 9. RAM Capacity (Standardized to numeric in GB)
    def parse_ram(val: Any) -> float:
        if pd.isna(val) or val is None:
            return np.nan
        val_str = str(val).upper()
        # Check TB (e.g., 1 TB RAM)
        tb_match = re.search(r"(\d+)\s*TB", val_str)
        if tb_match:
            return float(tb_match.group(1)) * 1024.0
        # Check GB (e.g., 16 GB DDR4, 8 GB Unified Memory)
        gb_match = re.search(r"(\d+)\s*GB", val_str)
        if gb_match:
            return float(gb_match.group(1))
        return np.nan

    cleaned_df["RAM"] = df["RAM"].apply(parse_ram)

    # 10. Storage Capacity (Unified numeric GB) & Storage Type (with title fallback for Chromebooks)
    def parse_storage(row: pd.Series) -> Tuple[float, str]:
        val = row.get("Storage_Raw")
        val_str = "" if pd.isna(val) or val is None else str(val).upper()

        # Dual storage detection (e.g., 256 GB SSD + 1 TB HDD)
        has_ssd = "SSD" in val_str
        has_hdd = "HDD" in val_str
        has_emmc = "EMMC" in val_str
        has_ufs = "UFS" in val_str

        ssd_matches = re.findall(r"(\d+)\s*(GB|TB)\s*SSD", val_str)
        hdd_matches = re.findall(r"(\d+)\s*(GB|TB)\s*HDD", val_str)

        if ssd_matches or hdd_matches:
            total_cap = 0.0
            for cap, unit in ssd_matches:
                total_cap += float(cap) * (1024.0 if unit == "TB" else 1.0)
            for cap, unit in hdd_matches:
                total_cap += float(cap) * (1024.0 if unit == "TB" else 1.0)
            st_type = "SSD + HDD" if (has_ssd and has_hdd) else ("SSD" if has_ssd else "HDD")
            return total_cap, st_type

        # Single storage parse
        tb_match = re.search(r"(\d+)\s*TB", val_str)
        gb_match = re.search(r"(\d+)\s*GB", val_str)

        total_cap = np.nan
        if tb_match:
            total_cap = float(tb_match.group(1)) * 1024.0
        elif gb_match:
            total_cap = float(gb_match.group(1))

        if pd.notna(total_cap):
            if has_ssd:
                st_type = "SSD"
            elif has_hdd:
                st_type = "HDD"
            elif has_emmc:
                st_type = "eMMC"
            elif has_ufs:
                st_type = "UFS"
            else:
                st_type = "SSD"
            return total_cap, st_type

        # Fallback: Extract eMMC / SSD storage directly from Product Name (e.g., Chromebooks)
        title_str = str(row.get("Product Name", "")).upper()
        t_emmc = re.search(r"(\d+)\s*(GB|TB)\s*(?:EMMC|STORAGE)", title_str)
        t_ssd = re.search(r"(\d+)\s*(GB|TB)\s*SSD", title_str)
        t_match = t_emmc or t_ssd
        if t_match:
            cap_val = float(t_match.group(1))
            unit_val = t_match.group(2)
            total_cap = cap_val * (1024.0 if unit_val == "TB" else 1.0)
            st_type = "eMMC" if t_emmc else "SSD"
            return total_cap, st_type

        return np.nan, "Not Specified"

    storage_tuples = df.apply(parse_storage, axis=1)
    cleaned_df["Storage"] = [t[0] for t in storage_tuples]
    cleaned_df["Storage Type"] = [t[1] for t in storage_tuples]

    # 11. Graphics (Integrated vs Dedicated) & GPU (Specific Model)
    def parse_graphics_and_gpu(row: pd.Series) -> Tuple[str, str]:
        raw_gfx = str(row.get("Graphics", "")).strip()
        raw_gpu = str(row.get("GPU", "")).strip()
        combined = f"{raw_gfx} {raw_gpu}".strip()

        if not combined or combined.lower() in ["none", "nan", "not specified"]:
            return "Integrated / Not Specified", "Not Specified"

        # Check for dedicated GPUs
        dedicated_keywords = ["nvidia", "geforce", "rtx", "gtx", "dedicated", "arc a", "radeon rx"]
        is_dedicated = any(k in combined.lower() for k in dedicated_keywords)

        # Extract clean GPU name
        gpu_model = "Not Specified"
        gpu_match = re.search(
            r"(NVIDIA\s+GeForce\s+(?:RTX|GTX)\s+\d+(?:\s*Ti|\s*Super)?|"
            r"AMD\s+Radeon\s+(?:RX\s+\d+\w*|\d+M)?|"
            r"Intel\s+(?:Iris\s+Xe|Arc\s+\w+|UHD\s+Graphics)|"
            r"Apple\s+M\d+\s+\d+-Core\s+GPU|"
            r"Qualcomm\s+Adreno)",
            combined,
            re.IGNORECASE,
        )
        if gpu_match:
            gpu_model = gpu_match.group(1).strip()
        elif raw_gpu and raw_gpu.lower() not in ["none", "nan"]:
            gpu_model = raw_gpu

        gfx_type = "Dedicated" if is_dedicated else "Integrated"
        return gfx_type, gpu_model

    gfx_tuples = df.apply(parse_graphics_and_gpu, axis=1)
    cleaned_df["Graphics"] = [t[0] for t in gfx_tuples]
    cleaned_df["GPU"] = [t[1] for t in gfx_tuples]

    # 12. Display Size (Numeric in Inches)
    def parse_display(val: Any) -> float:
        if pd.isna(val) or val is None:
            return np.nan
        val_str = str(val)
        # Search for inch specification (e.g., 15.6 inch, 14")
        inch_match = re.search(r"(\d+\.?\d*)\s*(?:inch|\'|\"|\-inch)", val_str, re.IGNORECASE)
        if inch_match:
            return float(inch_match.group(1))
        # Fallback to cm conversion (e.g., 39.62 cm)
        cm_match = re.search(r"(\d+\.?\d*)\s*cm", val_str, re.IGNORECASE)
        if cm_match:
            return round(float(cm_match.group(1)) / 2.54, 1)
        return np.nan

    cleaned_df["Display Size"] = df["Display Size"].apply(parse_display)

    # 13. Availability
    cleaned_df["Availability"] = (
        df["Availability"]
        .fillna("In Stock")
        .apply(lambda x: "Out of Stock" if "out" in str(x).lower() else "In Stock")
    )

    # 14. Touchscreen (Yes / No)
    cleaned_df["Touchscreen"] = (
        df["Touchscreen"]
        .fillna("No")
        .apply(lambda x: "Yes" if str(x).strip().lower() in ["yes", "true", "1"] else "No")
    )

    # 15. Operating System
    def clean_os(val: Any) -> str:
        if pd.isna(val) or val is None or str(val).lower() in ["none", "nan"]:
            return "Not Specified"
        v = str(val).lower()
        if "windows 11" in v:
            return "Windows 11 Home" if "home" in v else ("Windows 11 Pro" if "pro" in v else "Windows 11")
        elif "windows 10" in v:
            return "Windows 10"
        elif "mac" in v or "os x" in v:
            return "macOS"
        elif "chrome" in v:
            return "Chrome OS"
        elif "dos" in v:
            return "DOS"
        elif "linux" in v or "ubuntu" in v:
            return "Linux"
        return str(val).strip()

    cleaned_df["Operating System"] = df["Operating System"].apply(clean_os)

    # 16. Laptop Use
    cleaned_df["Laptop Use"] = df["Laptop Use"].fillna("Student / Everyday")

    # 17. Text Sanitation (Clean HTML entities, newlines, and excess spaces)
    text_columns = [
        "Product Name", "Brand", "Laptop Type", "Processor", "Storage Type",
        "Graphics", "GPU", "Availability", "Touchscreen", "Operating System", "Laptop Use"
    ]
    for col in text_columns:
        cleaned_df[col] = (
            cleaned_df[col]
            .astype(str)
            .str.replace(r"[\t\n\r]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace({"nan": "Not Specified", "": "Not Specified"})
        )

    # 18. Quality & Genuine Laptop Filtering Pipeline
    pre_filter_count = len(cleaned_df)
    logging.info(f"Applying quality & genuine laptop filters (Pre-filter count: {pre_filter_count} rows)...")

    # A. Exclude accessory keywords
    acc_pattern = (
        r"battery|adapter|charger|skin|cover|sleeve|case|screen guard|tempered glass|"
        r"trackpad|mouse|cable|usb light|sticker|keyboard skin"
    )
    is_not_accessory = ~cleaned_df["Product Name"].str.contains(acc_pattern, case=False, na=False)

    # B. Exclude refurbished/used & desktop/all-in-one devices
    refurb_pattern = r"\brefurb\b|\brenewed\b|\bpre-owned\b|\bsecond hand\b"
    is_not_refurb = ~cleaned_df["Product Name"].str.contains(refurb_pattern, case=False, na=False)

    desktop_pattern = r"\ball-in-one\b|\baio\b|\bdesktop\b|\btower pc\b"
    is_not_desktop = ~cleaned_df["Product Name"].str.contains(desktop_pattern, case=False, na=False)

    # C. Hardware sanity requirements (real laptops require RAM and CPU)
    has_valid_ram = cleaned_df["RAM"].notna() & (cleaned_df["RAM"] > 0)
    has_valid_processor = cleaned_df["Processor"] != "Not Specified"
    has_valid_screen = cleaned_df["Display Size"].isna() | (cleaned_df["Display Size"].between(10.0, 18.5))

    # D. Price & inventory sanity (Price >= 12,000 INR, Price <= MRP)
    has_valid_price = cleaned_df["Price"].notna() & (cleaned_df["Price"] >= 12000.0)
    cleaned_df["MRP"] = cleaned_df[["Price", "MRP"]].max(axis=1)

    # Combine all filters
    genuine_laptop_mask = (
        is_not_accessory
        & is_not_refurb
        & is_not_desktop
        & has_valid_ram
        & has_valid_processor
        & has_valid_screen
        & has_valid_price
    )

    cleaned_df = cleaned_df[genuine_laptop_mask].copy()
    logging.info(f"Quality filtering complete: Removed {pre_filter_count - len(cleaned_df)} non-laptop / invalid records.")

    # 19. Deduplication by Product Name and Brand
    initial_rows = len(cleaned_df)
    cleaned_df.drop_duplicates(subset=["Product Name", "Brand"], keep="first", inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)
    logging.info(f"Deduplication complete: Removed {initial_rows - len(cleaned_df)} duplicate listings.")

    # 20. Ensure exact column sequence of all 19 target columns
    TARGET_COLUMNS = [
        "Product Name",
        "Brand",
        "Laptop Type",
        "Price",
        "Discount",
        "MRP",
        "Rating",
        "Review Count",
        "Processor",
        "RAM",
        "Storage",
        "Storage Type",
        "Graphics",
        "Display Size",
        "GPU",
        "Availability",
        "Touchscreen",
        "Operating System",
        "Laptop Use",
    ]

    cleaned_df = cleaned_df[TARGET_COLUMNS]
    return cleaned_df


def validate_and_summarize(df: pd.DataFrame) -> bool:
    """
    Print diagnostic metrics, data types, and null distributions.
    Validates that the dataset conforms to Power BI requirements (no mixed types).
    """
    print("\n" + "=" * 70)
    print("        DATASET VALIDATION & POWER BI READINESS SUMMARY        ")
    print("=" * 70)
    print(f"Total Rows:    {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("-" * 70)

    summary_records = []
    has_mixed_types = False

    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = round((null_count / len(df) * 100), 2) if len(df) > 0 else 0.0

        # Check for mixed data types in non-null values
        types_in_col = {type(x).__name__ for x in df[col].dropna()}
        if len(types_in_col) > 1:
            has_mixed_types = True

        first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"

        summary_records.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Null Count": null_count,
            "Null %": f"{null_pct}%",
            "Unique Types": ", ".join(types_in_col) if types_in_col else "None",
            "Sample Value": str(first_valid)[:30],
        })

    summary_table = pd.DataFrame(summary_records)
    print(summary_table.to_string(index=False))
    print("-" * 70)

    if has_mixed_types:
        logging.warning("DATA QUALITY WARNING: Mixed types detected in columns. Resolve before Power BI import.")
    else:
        logging.info("DATA QUALITY CHECK PASSED: Zero mixed data types found. Ready for Power BI.")

    print("=" * 70 + "\n")
    return not has_mixed_types


def export_csv(df: pd.DataFrame, filename: str = "flipkart_laptops.csv") -> str:
    """
    Export cleaned DataFrame to UTF-8 with BOM (utf-8-sig) CSV.
    Guarantees seamless import into Power BI and Excel without currency or character distortion.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Successfully exported {len(df)} rows to Power BI CSV: {output_path}")
    return output_path


def main():
    """Main execution function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Scrape Flipkart laptops and generate a Power BI ready CSV dataset."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=5000,
        help="Target number of laptop listings to harvest (default: 5000).",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        default=False,
        help="Enable deep scraping of individual product pages (slower, extracts detailed JSON specs).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run quick test mode (harvests ~24-48 items to verify pipeline and CSV export).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flipkart_laptops.csv",
        help="Output CSV filename (default: flipkart_laptops.csv).",
    )

    args = parser.parse_args()

    target_count = 30 if args.test else args.target
    logging.info(f"Running scraper in {'TEST' if args.test else 'PRODUCTION'} mode.")

    # Step 1: Scrape raw listings
    raw_df = scrape_flipkart(
        target_count=target_count,
        deep_scrape=args.deep,
        checkpoint_file="checkpoint_laptops.csv",
    )

    if raw_df.empty:
        logging.error("No data could be extracted from Flipkart. Pipeline halted.")
        return

    # Step 2: Clean, standardize, and format for Power BI
    clean_df = clean_data(raw_df)

    # Step 3: Validate data quality and print Power BI readiness summary
    validate_and_summarize(clean_df)

    # Step 4: Export to UTF-8 BOM CSV
    export_csv(clean_df, args.output)


if __name__ == "__main__":
    main()
