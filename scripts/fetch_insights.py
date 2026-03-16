#!/usr/bin/env python3
"""
Hourly cron job to fetch Instagram post insights into SQLite.

- Fetches ALL posts from March 2026 onwards
- Every hour stores a new snapshot (never overwrites old data)
- You can query the DB to see hourly timeline + deltas for any post

Usage:
    python scripts/fetch_insights.py                    # Run once
    python scripts/fetch_insights.py --since 2026-02-01 # Custom start date

Cron (every hour):
    0 * * * * cd /home/nicole/MyGithub/ig-mcp && /usr/bin/python3 scripts/fetch_insights.py >> logs/cron_insights.log 2>&1
"""

import argparse
import asyncio
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.config import get_settings
from src.instagram_client import InstagramClient

# ── Config ───────────────────────────────────────────────────────

DB_PATH = PROJECT_ROOT / "data" / "insights.db"
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_SINCE = "2026-03-01"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "cron_insights.log"),
    ],
)
logger = logging.getLogger(__name__)

# Metrics per post type
METRICS_IMAGE = ["impressions", "reach", "saved", "likes", "comments", "shares"]
METRICS_VIDEO = ["impressions", "reach", "saved", "likes", "comments", "shares",
                 "video_views", "plays"]


# ── Database ─────────────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Each row = one metric for one post at one point in time
    # fetched_at is the hourly snapshot time — never overwritten
    c.execute("""
        CREATE TABLE IF NOT EXISTS post_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT NOT NULL,
            media_id TEXT NOT NULL,
            media_type TEXT,
            caption TEXT,
            permalink TEXT,
            posted_at TEXT,
            metric_name TEXT NOT NULL,
            metric_value INTEGER,
            UNIQUE(fetched_at, media_id, metric_name)
        )
    """)

    # Breakdown data (follower/non-follower, traffic source)
    c.execute("""
        CREATE TABLE IF NOT EXISTS post_insights_breakdown (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT NOT NULL,
            media_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            breakdown_dimension TEXT NOT NULL,
            breakdown_key TEXT NOT NULL,
            breakdown_value INTEGER,
            UNIQUE(fetched_at, media_id, metric_name, breakdown_dimension, breakdown_key)
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_pi_media ON post_insights(media_id, fetched_at)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_bd_media ON post_insights_breakdown(media_id, fetched_at)")
    conn.commit()
    return conn


def store_insights(conn, fetched_at, media_id, media_type, caption, permalink,
                   posted_at, insights):
    c = conn.cursor()
    for item in insights:
        name = item.get("name", "")
        values = item.get("values", [])
        if values:
            value = values[0].get("value", 0)
            if isinstance(value, int):
                c.execute("""
                    INSERT OR REPLACE INTO post_insights
                    (fetched_at, media_id, media_type, caption, permalink, posted_at,
                     metric_name, metric_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (fetched_at, media_id, media_type, caption[:200], permalink,
                      posted_at, name, value))
    conn.commit()


def store_breakdowns(conn, fetched_at, media_id, breakdowns):
    c = conn.cursor()
    for item in breakdowns:
        name = item.get("name", "")
        total_value = item.get("total_value", {})
        if not total_value:
            continue
        for bd in total_value.get("breakdowns", []):
            dimension = "_".join(bd.get("dimension_keys", ["unknown"]))
            for result in bd.get("results", []):
                key = "_".join(result.get("dimension_values", ["unknown"]))
                value = result.get("value", 0)
                c.execute("""
                    INSERT OR REPLACE INTO post_insights_breakdown
                    (fetched_at, media_id, metric_name, breakdown_dimension,
                     breakdown_key, breakdown_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (fetched_at, media_id, name, dimension, key, value))
    conn.commit()


# ── Fetch Logic ──────────────────────────────────────────────────

async def fetch_and_store(since_date: str = DEFAULT_SINCE):
    settings = get_settings()
    client = InstagramClient(settings)
    conn = init_db()

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    since_dt = datetime.strptime(since_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    try:
        await client.initialize()

        # Fetch posts newest-first, stop when we pass since_date
        logger.info(f"Fetching posts since {since_date}...")
        posts = await client.get_media_posts(limit=50)
        target_posts = [p for p in posts if p.timestamp and p.timestamp >= since_dt]

        logger.info(f"Found {len(target_posts)} posts since {since_date}")

        for i, post in enumerate(target_posts):
            media_id = post.id
            media_type = post.media_type.value if post.media_type else "IMAGE"
            caption = post.caption or ""
            permalink = post.permalink or ""
            posted_at = post.timestamp.isoformat() if post.timestamp else ""
            preview = caption[:50].replace('\n', ' ')

            logger.info(f"  [{i+1}/{len(target_posts)}] {posted_at[:10]} | {preview}...")

            # Basic metrics
            try:
                metrics = METRICS_VIDEO if media_type in ("VIDEO", "REELS") else METRICS_IMAGE
                data = await client._make_request(
                    "GET", f"{media_id}/insights",
                    params={"metric": ",".join(metrics)},
                )
                store_insights(conn, fetched_at, media_id, media_type,
                               caption, permalink, posted_at,
                               data.get("data", []))
            except Exception as e:
                logger.warning(f"    Metrics failed: {e}")

            # Follower vs non-follower
            try:
                bd_data = await client._make_request(
                    "GET", f"{media_id}/insights",
                    params={
                        "metric": "reach",
                        "metric_type": "total_value",
                        "breakdown": "follow_type",
                    },
                )
                store_breakdowns(conn, fetched_at, media_id, bd_data.get("data", []))
            except Exception as e:
                logger.warning(f"    Follower breakdown failed: {e}")

            # Traffic source
            try:
                src_data = await client._make_request(
                    "GET", f"{media_id}/insights",
                    params={
                        "metric": "reach",
                        "metric_type": "total_value",
                        "breakdown": "media_product_type",
                    },
                )
                store_breakdowns(conn, fetched_at, media_id, src_data.get("data", []))
            except Exception as e:
                logger.warning(f"    Traffic source failed: {e}")

        # Summary
        c = conn.cursor()
        c.execute("""
            SELECT metric_name, SUM(metric_value)
            FROM post_insights WHERE fetched_at = ?
            GROUP BY metric_name ORDER BY metric_name
        """, (fetched_at,))
        logger.info("── Summary ──")
        for name, total in c.fetchall():
            logger.info(f"  {name}: {total:,}")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        conn.close()
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="Fetch Instagram post insights")
    parser.add_argument("--since", type=str, default=DEFAULT_SINCE,
                        help=f"Fetch posts from this date onwards (default: {DEFAULT_SINCE})")
    args = parser.parse_args()

    asyncio.run(fetch_and_store(args.since))


if __name__ == "__main__":
    main()
