"""Fetch top artists per genre from Last.fm, enrich with MusicBrainz, write CSV.

Pipeline (per genre defined in ``configs/artist_genres.yaml``):
  1. Last.fm ``tag.gettopartists`` -> ranked list of candidates.
  2. Last.fm ``artist.getinfo``    -> listeners, playcount, canonical MBID.
  3. MusicBrainz ``/artist/{mbid}``-> type, gender, country.
     Fallback: ``/artist/?query=artist:{name}`` if no MBID, keep hit if score >= 90.
  4. Filter by country (configurable allow-list).
  5. Keep the top-N surviving artists per genre (default N=20).
  6. Write one row per (artist x genre) to ``data/artists_raw_{YYYYMMDD}.csv``.

Usage:
  export LASTFM_API_KEY=...   # (or put it in .env)
  python utils/fetch_artists.py                     # default: top 20 per genre
  python utils/fetch_artists.py --top-n 50
  python utils/fetch_artists.py --config configs/artist_genres.yaml --out-dir data/

MusicBrainz rate limit is 1 req/sec and is enforced globally.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

LASTFM_URL = "https://ws.audioscrobbler.com/2.0/"
MB_URL = "https://musicbrainz.org/ws/2"
USER_AGENT = "lyrkl-artist-fetcher/0.1 ( https://github.com/ )"
DEFAULT_COUNTRIES = {"US", "GB", "CA", "AU", "IE", "NZ"}
MB_MIN_INTERVAL_S = 1.05
MB_SEARCH_MIN_SCORE = 90


def _lastfm_get(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any] | None:
    try:
        r = client.get(LASTFM_URL, params=params)
    except httpx.HTTPError as exc:
        print(f"  ! last.fm error ({params.get('method')}): {exc}", flush=True)
        return None
    if r.status_code != 200:
        print(
            f"  ! last.fm {params.get('method')} -> {r.status_code}: {r.text[:120]}",
            flush=True,
        )
        return None
    return r.json()


def lastfm_top_artists(
    client: httpx.Client, api_key: str, tag: str, limit: int
) -> list[dict[str, Any]]:
    data = _lastfm_get(
        client,
        {
            "method": "tag.gettopartists",
            "tag": tag,
            "limit": limit,
            "api_key": api_key,
            "format": "json",
        },
    )
    if not data:
        return []
    return data.get("topartists", {}).get("artist", []) or []


def lastfm_artist_info(
    client: httpx.Client, api_key: str, name: str, mbid: str | None
) -> dict[str, Any] | None:
    params: dict[str, Any] = {
        "method": "artist.getinfo",
        "artist": name,
        "api_key": api_key,
        "format": "json",
        "autocorrect": 1,
    }
    if mbid:
        params["mbid"] = mbid
    data = _lastfm_get(client, params)
    if not data:
        return None
    return data.get("artist")


class _MusicBrainz:
    """Thin MusicBrainz wrapper that enforces the 1 req/sec global rate limit."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self._last_call_ts = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < MB_MIN_INTERVAL_S:
            time.sleep(MB_MIN_INTERVAL_S - elapsed)
        self._last_call_ts = time.monotonic()

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        self._wait()
        try:
            r = self._client.get(
                f"{MB_URL}{path}",
                params=params,
                headers={"User-Agent": USER_AGENT},
            )
        except httpx.HTTPError as exc:
            print(f"  ! musicbrainz error ({path}): {exc}", flush=True)
            return None
        if r.status_code == 503:
            # MB asks us to back off; wait and retry once
            time.sleep(2.0)
            return self._get(path, params)
        if r.status_code != 200:
            return None
        return r.json()

    def by_mbid(self, mbid: str) -> dict[str, Any] | None:
        return self._get(f"/artist/{mbid}", {"fmt": "json"})

    def search_by_name(self, name: str) -> dict[str, Any] | None:
        data = self._get(
            "/artist/",
            {"query": f'artist:"{name}"', "fmt": "json", "limit": 1},
        )
        if not data:
            return None
        hits = data.get("artists") or []
        if not hits:
            return None
        top = hits[0]
        if int(top.get("score", 0)) < MB_SEARCH_MIN_SCORE:
            return None
        return top


def _country_code(mb: dict[str, Any]) -> str | None:
    area = mb.get("area") or {}
    codes = area.get("iso-3166-1-codes") or []
    return codes[0] if codes else None


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="configs/artist_genres.yaml",
        help="YAML with per-genre tag + era windows (default: configs/artist_genres.yaml)",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of artists kept per genre after filtering (default: 20)",
    )
    ap.add_argument(
        "--fetch-limit",
        type=int,
        default=150,
        help="Last.fm candidates per genre before filtering (default: 150)",
    )
    ap.add_argument(
        "--out-dir",
        default="data",
        help="Directory for the output CSV (default: data/)",
    )
    ap.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Override allowed country codes from config",
    )
    args = ap.parse_args()

    load_dotenv(override=False)
    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        print(
            "LASTFM_API_KEY not set. Add it to .env or export it before running.",
            file=sys.stderr,
        )
        return 2

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = _load_config(cfg_path)

    allowed_countries = set(
        args.countries or cfg.get("countries") or DEFAULT_COUNTRIES
    )
    genres = cfg.get("genres") or {}
    if not genres:
        print("No genres defined in config.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    out_path = out_dir / f"artists_raw_{today.strftime('%Y%m%d')}.csv"

    # Artist-level caches (shared across genres to avoid duplicate API work)
    info_cache: dict[str, dict[str, Any]] = {}
    mb_cache: dict[str, dict[str, Any]] = {}

    rows: list[dict[str, Any]] = []

    with httpx.Client(timeout=30.0) as client:
        mb = _MusicBrainz(client)

        for genre, meta in genres.items():
            tag = meta["tag"]
            print(f"\n=== {genre}  (tag={tag!r}) ===", flush=True)

            candidates = lastfm_top_artists(client, api_key, tag, args.fetch_limit)
            print(f"  last.fm: {len(candidates)} candidates", flush=True)

            kept: list[dict[str, Any]] = []
            for raw in candidates:
                if len(kept) >= args.top_n:
                    break

                name = raw.get("name")
                if not name:
                    continue
                mbid = (raw.get("mbid") or "").strip()

                info_key = mbid or f"name:{name.lower()}"
                info = info_cache.get(info_key)
                if info is None:
                    info = lastfm_artist_info(client, api_key, name, mbid) or {}
                    info_cache[info_key] = info
                if not info:
                    continue

                mbid = (info.get("mbid") or mbid or "").strip()
                stats = info.get("stats") or {}
                listeners = int(stats.get("listeners") or 0)
                playcount = int(stats.get("playcount") or 0)
                canonical_name = info.get("name") or name

                mb_key = mbid or f"name:{canonical_name.lower()}"
                mb_data = mb_cache.get(mb_key)
                if mb_data is None:
                    if mbid:
                        mb_data = mb.by_mbid(mbid) or {}
                    else:
                        mb_data = mb.search_by_name(canonical_name) or {}
                        if mb_data and not mbid:
                            mbid = mb_data.get("id") or ""
                    mb_cache[mb_key] = mb_data
                if not mb_data:
                    continue

                country = _country_code(mb_data)
                if country not in allowed_countries:
                    continue

                kept.append(
                    {
                        "mbid": mbid,
                        "name": canonical_name,
                        "type": mb_data.get("type") or "",
                        "gender": mb_data.get("gender") or "",
                        "country": country,
                        "listeners": listeners,
                        "playcount": playcount,
                    }
                )

            print(f"  kept: {len(kept)} after country filter", flush=True)
            for rank, a in enumerate(kept, start=1):
                rows.append(
                    {
                        "mbid": a["mbid"],
                        "name": a["name"],
                        "type": a["type"],
                        "gender": a["gender"],
                        "country": a["country"],
                        "genre": genre,
                        "genre_rank": rank,
                        "lastfm_listeners": a["listeners"],
                        "lastfm_playcount": a["playcount"],
                        "retrieved_date": today.isoformat(),
                    }
                )

    fieldnames = [
        "mbid",
        "name",
        "type",
        "gender",
        "country",
        "genre",
        "genre_rank",
        "lastfm_listeners",
        "lastfm_playcount",
        "retrieved_date",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
