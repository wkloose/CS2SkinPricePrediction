import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

STEAM_API_URL = "https://steamcommunity.com/market/priceoverview/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def load_items_config(config_path: str = "configs/items.json") -> List[str]:
    """Membaca daftar nama skin CS2 dari file konfigurasi JSON."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        items = json.load(file)
    logging.info("Berhasil memuat %d item dari %s", len(items), config_path)
    return items


def fetch_item_price(
    item_name: str,
    max_retries: int = 3,
    delay_seconds: float = 2.5
) -> Optional[Dict[str, Any]]:
    """
    Mengambil data harga skin dari steam market API dengan mekanisme retry.
    """
    params = {
        "appid": 730,
        "currency": 10,  # IDR (Rupiah)
        "market_hash_name": item_name
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                STEAM_API_URL,
                params=params,
                headers=HEADERS,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return {
                        "market_hash_name": item_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "lowest_price": data.get("lowest_price", None),
                        "median_price": data.get("median_price", None),
                        "volume": data.get("volume", None),
                        "success": True
                    }
                logging.warning("API merespons success:false untuk %s", item_name)
                return None

            elif response.status_code == 429:
                wait_time = delay_seconds * (2 ** attempt)
                logging.warning(
                    "Terkena rate limit (429) untuk %s. menunggu %.1f detik...",
                    item_name, wait_time
                )
                time.sleep(wait_time)
            else:
                logging.error(
                    "Gagal mengambil %s: Status Code %d",
                    item_name, response.status_code
                )
                return None

        except requests.RequestException as err:
            logging.error("Error koneksi pada %s (percobaan %d): %s", item_name, attempt, err)
            time.sleep(delay_seconds)

    return None


def run_ingestion(
    config_path: str = "configs/items.json",
    output_dir: str = "data/raw",
    throttle_delay: float = 2.0
) -> str:
    """
    Menjalankan seluruh pipeline ingestion secara non-destruktif.
    """
    os.makedirs(output_dir, exist_ok=True)
    items = load_items_config(config_path)

    collected_records: List[Dict[str, Any]] = []
    logging.info("Memulai proses ingestion data untuk %d items...", len(items))

    for idx, item in enumerate(items, start=1):
        logging.info("[%d/%d] Mengambil data: %s", idx, len(items), item)
        record = fetch_item_price(item, delay_seconds=throttle_delay)
        if record:
            collected_records.append(record)
        time.sleep(throttle_delay)

    if not collected_records:
        raise RuntimeError("Tidak ada data yang berhasil diambil dari steam API.")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"cs2_raw_{timestamp_str}.csv")
    output_json = os.path.join(output_dir, f"cs2_raw_{timestamp_str}.json")

    # Non-destructive saving (CSV & JSON)
    df = pd.DataFrame(collected_records)
    df.to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as f_json:
        json.dump(collected_records, f_json, indent=2)

    logging.info("Ingestion selesai! data tersimpan di:")
    logging.info(" - CSV : %s (%d baris)", output_csv, len(df))
    logging.info(" - JSON: %s", output_json)
    return output_csv


if __name__ == "__main__":
    run_ingestion()
