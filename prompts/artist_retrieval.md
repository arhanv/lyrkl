## Artist List Retrieval Instructions

### Goal
Retrieve the top 100 English-language artists for each of the following genres: *pop, rock, hip-hop, r&b, country*. Enrich each artist with metadata. Output a single versioned CSV.

---

### Step 1 — Get a Last.fm API Key
Register at https://www.last.fm/api/account/create (free, instant). You'll get a key immediately.

---

### Step 2 — Fetch Top Artists Per Genre from Last.fm

For each genre, call:

GET https://ws.audioscrobbler.com/2.0/
  ?method=tag.gettopartists
  &tag={genre}
  &limit=150
  &api_key={KEY}
  &format=json


Use these tag strings: ⁠ pop ⁠, ⁠ rock ⁠, ⁠ hip-hop ⁠, ⁠ rnb ⁠, ⁠ country ⁠

From each result, extract: ⁠ name ⁠, ⁠ mbid ⁠, ⁠ listeners ⁠ (if present in response — not always populated here; supplement in Step 4).

---

### Step 3 — Fetch Per-Artist Stats from Last.fm

For each artist returned above, call:

GET https://ws.audioscrobbler.com/2.0/
  ?method=artist.getinfo
  &artist={name}
  &mbid={mbid}
  &api_key={KEY}
  &format=json


Extract: ⁠ listeners ⁠, ⁠ playcount ⁠, ⁠ mbid ⁠ (use this to correct any missing MBIDs from Step 2).

---

### Step 4 — Enrich from MusicBrainz

For each artist with a valid ⁠ mbid ⁠, call:

GET https://musicbrainz.org/ws/2/artist/{mbid}?fmt=json


Rate limit: *1 request/second*. Extract: ⁠ type ⁠ (Person or Group), ⁠ gender ⁠ (Female/Male/null), ⁠ area.iso-3166-1-codes ⁠ (country code).

For artists missing an MBID, search by name:

GET https://musicbrainz.org/ws/2/artist/?query=artist:{name}&fmt=json

Take the top result if score ≥ 90.

---

### Step 5 — Filter

Keep only artists where ⁠ area ⁠ country code is in: ⁠ US, GB, CA, AU, IE, NZ ⁠. Drop artists ranked below 100 within their genre after filtering.

---

### Step 6 — Output CSV

Write one row per artist × genre combination (an artist appearing in multiple genres gets multiple rows). Columns:


mbid, name, type, gender, country, genre, genre_rank, lastfm_listeners, lastfm_playcount, retrieved_date


Set ⁠ retrieved_date ⁠ to today's ISO date. Save as ⁠ artists_raw_{YYYYMMDD}.csv ⁠.