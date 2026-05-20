#!/usr/bin/env python3
"""
Collect confirmatory dataset from an earlier time period.

Main dataset: Apr–Jul 2025 (CharacterAI), Jan 2024 (Replika)
Confirmatory: Jan 2024 – Mar 2025 (before the main bulk)

Subreddits: CharacterAI, replika
Target: ~50K comments
Source: Arctic Shift API (free, no auth)
"""

import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent.parent
OUT_DIR = PROJECT / "Data" / "confirmatory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://arctic-shift.photon-reddit.com"
RATE_LIMIT = 1.0

# Collect Jan 2024 – Mar 2025 for these subreddits
SUBREDDITS = ["CharacterAI", "replika"]
START_UTC = int(datetime(2024, 1, 1).timestamp())
END_UTC = int(datetime(2025, 3, 31).timestamp())

# Existing data IDs to exclude (avoid overlap)
EXISTING_IDS = set()


def load_existing_ids():
    """Load comment IDs from existing dataset to avoid duplicates."""
    global EXISTING_IDS
    main_path = PROJECT / "Data" / "processed" / "all_comments.parquet"
    if main_path.exists():
        df = pd.read_parquet(main_path, columns=['id'])
        EXISTING_IDS = set(df['id'].astype(str))
        logger.info(f"Loaded {len(EXISTING_IDS):,} existing IDs to exclude")


def fetch_comments(subreddit, after_utc, before_utc, limit=100):
    """Fetch a page of comments from Arctic Shift."""
    params = {
        "subreddit": subreddit,
        "after": int(after_utc),
        "before": int(before_utc),
        "limit": limit,
        "sort": "asc",
    }
    for attempt in range(3):
        try:
            r = requests.get(
                f"{BASE_URL}/api/comments/search",
                params=params,
                headers={"User-Agent": "Teen-AI-Companion-Research/1.0"},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep((attempt + 1) * 3)
    return []


def collect_subreddit(subreddit, start_utc, end_utc, output_path, max_comments=None):
    """Collect all comments for a subreddit in a time range."""
    logger.info(f"Collecting r/{subreddit}: "
                f"{datetime.fromtimestamp(start_utc)} → {datetime.fromtimestamp(end_utc)}")

    all_comments = []
    current_utc = start_utc
    seen_ids = set()
    empty_streak = 0

    with tqdm(desc=f"r/{subreddit}") as pbar:
        while current_utc < end_utc and empty_streak < 5:
            comments = fetch_comments(subreddit, current_utc, end_utc, limit=100)

            if not comments:
                empty_streak += 1
                current_utc += 86400
                continue

            empty_streak = 0
            max_ts = current_utc

            for c in comments:
                cid = c.get("id", "")
                if cid in seen_ids or cid in EXISTING_IDS:
                    continue
                seen_ids.add(cid)

                body = c.get("body", "")
                if not body or body in ("[deleted]", "[removed]"):
                    continue
                author = c.get("author", "")
                if not author or author in ("[deleted]", "AutoModerator"):
                    continue

                all_comments.append(c)
                pbar.update(1)

                ts = c.get("created_utc", 0)
                if isinstance(ts, (int, float)):
                    max_ts = max(max_ts, int(ts))

            current_utc = max_ts + 1

            if max_comments and len(all_comments) >= max_comments:
                logger.info(f"Hit max_comments={max_comments}")
                all_comments = all_comments[:max_comments]
                break

            time.sleep(RATE_LIMIT)

    if all_comments:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for c in all_comments:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(all_comments):,} comments → {output_path}")

    return all_comments


def build_parquet(raw_dir, output_path):
    """Convert raw JSONL files into a single processed parquet."""
    all_rows = []

    for jsonl in raw_dir.glob("*.jsonl"):
        logger.info(f"Processing {jsonl.name}...")
        with open(jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                all_rows.append({
                    'id': str(c.get('id', '')),
                    'author': c.get('author', ''),
                    'body': c.get('body', ''),
                    'created_utc': c.get('created_utc', 0),
                    'subreddit': c.get('subreddit', ''),
                    'link_id': c.get('link_id', ''),
                    'parent_id': c.get('parent_id', ''),
                    'score': c.get('score', 0),
                    'author_flair_text': c.get('author_flair_text', ''),
                })

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset='id')

    # Filter bots
    bot_patterns = ['bot', 'auto', 'moderator', 'reminder', 'transcriber']
    bot_mask = df['author'].str.lower().str.contains('|'.join(bot_patterns), na=False)
    df = df[~bot_mask]

    # Filter very short
    df = df[df['body'].str.len() >= 10]

    df.to_parquet(output_path, index=False)
    logger.info(f"Final confirmatory dataset: {len(df):,} comments → {output_path}")

    # Stats
    df['created_dt'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    logger.info(f"  Date range: {df['created_dt'].min()} → {df['created_dt'].max()}")
    logger.info(f"  Unique authors: {df['author'].nunique():,}")
    for sub in df['subreddit'].unique():
        n = len(df[df['subreddit'] == sub])
        logger.info(f"  r/{sub}: {n:,} comments")

    return df


def main():
    logger.info("=" * 60)
    logger.info("COLLECTING CONFIRMATORY DATASET")
    logger.info(f"Period: {datetime.fromtimestamp(START_UTC)} → {datetime.fromtimestamp(END_UTC)}")
    logger.info(f"Subreddits: {SUBREDDITS}")
    logger.info("=" * 60)

    load_existing_ids()

    raw_dir = OUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for sub in SUBREDDITS:
        output = raw_dir / f"{sub.lower()}_confirmatory.jsonl"
        if output.exists():
            with open(output) as f:
                n = sum(1 for _ in f)
            logger.info(f"Skipping r/{sub} — already collected ({n:,} comments)")
            continue
        collect_subreddit(sub, START_UTC, END_UTC, output)

    # Build processed parquet
    parquet_path = OUT_DIR / "confirmatory_comments.parquet"
    df = build_parquet(raw_dir, parquet_path)

    logger.info("=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info(f"Ready for scoring: {parquet_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
