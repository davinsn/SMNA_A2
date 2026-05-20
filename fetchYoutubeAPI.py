#
# COSC2671 Social Media and Network Analytics
# @author Hexu Chen, RMIT University, 2026
# @author Chenglong Ma, RMIT University, 2026
#
# Utility script to fetch YouTube data and save as a JSON file.
#
# Usage:
#   python fetchYoutubeData.py
#
# Make sure to set your API key in youtubeClient.py first!
#
# Modified and adapted by: Arnavv Vivek Khabile & Davin Subash Nair
# File: fetchYoutubeData.py
#
# Purpose:
# Fetches Formula 1 safety-related YouTube videos and comments
# from 2018 onwards for sentiment analysis, topic modelling,
# and network analysis.
#

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
from googleapiclient.errors import HttpError

from youtubeClient import youtubeClient

DetectorFactory.seed = 0


# ── Search queries ────────────────────────────────────────────────────────────

SEARCH_QUERIES = [
    "Formula 1 dangerous crashes",
    "Romain Grosjean Bahrain crash analysis",
    "F1 safety improvements",
    "Formula 1 controversial crashes",
    "Formula 1 crash aftermath discussion",
    "Romain Grosjean fire Bahrain 2020",
    "Lewis Hamilton crash 2021",
    "Max Verstappen crash 2021",
    "Zhou Guanyu crash Silverstone 2022",
    "Formula 1 safety car controversy",
]


# ── Keyword filter lists ──────────────────────────────────────────────────────

EXCLUDE_KEYWORDS = [
    "#shorts", "shorts", "meme", "edit",
    "funny moments", "gaming", "f1 game",
    "hotlap", "mod", "simulator",
    "reaction meme", "fan cam", "fancam",
    "tiktok", "compilation", "best overtakes",
    "top speed", "engine sound", "pit stop challenge",
    "driver ranking", "predictions", "career mode",
]

SAFETY_KEYWORDS = [
    "safety", "crash", "accident", "incident",
    "dangerous", "injury", "fire", "collision",
    "halo", "fia", "medical", "protection",
    "fatal", "survival", "barrier", "marshals",
]

ACCIDENT_KEYWORDS = [
    "zhou", "verstappen", "hamilton",
    "silverstone", "bahrain", "imola",
    "spa", "monaco", "crash analysis",
]

ANALYSIS_KEYWORDS = [
    "analysis", "breakdown", "documentary",
    "investigation", "review", "discussion",
    "reaction", "explained", "technical analysis",
]

FORMULA1_KEYWORDS = [
    "formula 1", "f1", "fia", "grand prix",
    "motorsport", "formula one",
]

STRONG_CONTEXT_KEYWORDS = [
    "race", "driver", "safety", "fia",
    "crash", "accident", "barrier",
    "halo", "track", "medical car",
    "red flag", "investigation",
]

F1_REQUIRED_KEYWORDS = [
    "formula 1", "f1", "grand prix",
    "crash", "accident", "safety",
    "driver", "fia", "motorsport",
]

OFFICIAL_CHANNEL_HINTS = [
    "formula 1", "f1", "fia",
    "sky sports", "espn",
    "motorsport", "the race",
    "autosport", "bbc sport",
]

# Comments containing ONLY these patterns are considered junk
JUNK_COMMENT_PATTERNS = [
    r"^[\s\d]+$",                        # only numbers/whitespace
    r"^(.)\1{4,}$",                      # repeated single character e.g. "lolololol"
    r"^(ha|lol|lmao|haha|😂|🤣){2,}$",  # pure laugh filler
    r"^\W+$",                            # only punctuation/symbols
]


# ── Text cleaning ─────────────────────────────────────────────────────────────

def strip_emojis_and_symbols(text):
    """Keep only ASCII letters, digits, basic punctuation, and spaces."""
    if not text:
        return ""
    # Remove emoji and non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove anything that isn't a letter, digit, common punctuation, or space
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", " ", text)
    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_junk_comment(text):
    """Return True if the comment carries no useful content."""
    t = text.strip().lower()

    # Too short to be meaningful after cleaning
    if len(t) < 10:
        return True

    # Fewer than 3 words
    if len(t.split()) < 3:
        return True

    # Matches a known junk pattern
    for pattern in JUNK_COMMENT_PATTERNS:
        if re.fullmatch(pattern, t, re.IGNORECASE):
            return True

    return False


# ── Relevance scoring ─────────────────────────────────────────────────────────

def normalize_text(text):
    text = (text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text, keywords):
    return any(kw in text for kw in keywords)


def calculate_relevance_score(title, description="", channel_title=""):

    t = normalize_text(title)
    d = normalize_text(description)
    c = normalize_text(channel_title)

    combined = f"{t} {d}"

    score = 0

    # Must mention Formula 1 in some form
    if contains_any(combined, FORMULA1_KEYWORDS):
        score += 4
    else:
        return -100

    # Hard reject unwanted content types
    if contains_any(combined, EXCLUDE_KEYWORDS):
        return -100

    # Reward safety discussion
    if contains_any(combined, SAFETY_KEYWORDS):
        score += 5

    # Reward accident discussion
    if contains_any(combined, ACCIDENT_KEYWORDS):
        score += 4

    # Reward analytical content
    if contains_any(combined, ANALYSIS_KEYWORDS):
        score += 4

    # Reward strong contextual relevance (capped at 5)
    hits = sum(1 for kw in STRONG_CONTEXT_KEYWORDS if kw in combined)
    score += min(hits, 5)

    # Reward known reliable channels
    if contains_any(c, OFFICIAL_CHANNEL_HINTS):
        score += 3

    return score


def assign_video_category(title, description=""):

    combined = normalize_text(f"{title} {description}")

    if contains_any(combined, SAFETY_KEYWORDS):
        return "safety_discussion"
    elif contains_any(combined, ACCIDENT_KEYWORDS):
        return "accident_analysis"
    elif contains_any(combined, ANALYSIS_KEYWORDS):
        return "technical_analysis"

    return "other"


# ── ISO 8601 duration parser ──────────────────────────────────────────────────

def parse_iso8601_duration(s):

    m = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        s or ""
    )

    if not m:
        return 0

    return (
        int(m.group(1) or 0) * 3600
        + int(m.group(2) or 0) * 60
        + int(m.group(3) or 0)
    )


# ── YouTube API helpers ───────────────────────────────────────────────────────

def search_videos_for_query(
    client,
    query,
    max_videos=50,
    published_after=None
):

    results = []
    next_page_token = None

    while len(results) < max_videos:

        batch = min(50, max_videos - len(results))

        resp = client.search().list(
            q=query,
            part="snippet",
            type="video",
            videoDuration="medium",
            order="relevance",
            maxResults=batch,
            pageToken=next_page_token,
            publishedAfter=published_after,
        ).execute()

        results.extend(resp.get("items", []))

        next_page_token = resp.get("nextPageToken")

        if not next_page_token:
            break

    return results


def deduplicate(items):

    seen = set()
    out = []

    for item in items:
        vid = item["id"]["videoId"]
        if vid not in seen:
            seen.add(vid)
            out.append(item)

    return out


def rank_and_filter(items, min_score=5, max_videos=60):
    """
    Lenient filter: min_score lowered to 5 (was 7) so more borderline
    F1 safety videos pass through. Hard exclusions (EXCLUDE_KEYWORDS)
    still apply inside calculate_relevance_score.
    """
    scored = []

    for item in items:

        s = item.get("snippet", {})

        score = calculate_relevance_score(
            s.get("title", ""),
            s.get("description", ""),
            s.get("channelTitle", "")
        )

        if score >= min_score:
            scored.append((score, item))

    scored.sort(
        key=lambda x: (
            x[0],
            x[1]["snippet"].get("publishedAt", "")
        ),
        reverse=True
    )

    return [item for _, item in scored[:max_videos]]


def get_video_details(client, video_ids):

    if not video_ids:
        return []

    all_items = []

    for i in range(0, len(video_ids), 50):

        chunk = video_ids[i:i + 50]

        resp = client.videos().list(
            id=",".join(chunk),
            part="snippet,statistics,contentDetails",
        ).execute()

        all_items.extend(resp.get("items", []))

    videos = []

    for item in all_items:

        duration = parse_iso8601_duration(
            item.get("contentDetails", {}).get("duration", "PT0S")
        )

        # Skip very short videos
        if duration < 180:
            continue

        snippet = item.get("snippet", {})
        stats   = item.get("statistics", {})
        title   = snippet.get("title", "")
        desc    = snippet.get("description", "")

        videos.append({
            "title":        title,
            "videoId":      item["id"],
            "channelTitle": snippet.get("channelTitle", ""),
            "publishedAt":  snippet.get("publishedAt", ""),
            "viewCount":    int(stats.get("viewCount", 0)),
            "likeCount":    int(stats.get("likeCount", 0)),
            "commentCount": int(stats.get("commentCount", 0)),
            "durationSecs": duration,
            "category":     assign_video_category(title, desc),
            "comments":     [],
        })

    return videos


# ── Comment processing ────────────────────────────────────────────────────────

def process_comment(raw_text):
    """
    Detect language, translate non-English to English,
    then strip emojis/symbols from the result.
    Returns (cleaned_text, original_lang, was_translated).
    """
    try:
        lang = detect(raw_text)
    except LangDetectException:
        lang = "unknown"

    translated_text = raw_text
    was_translated  = False

    if lang != "en" and raw_text.strip():
        try:
            result = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(raw_text)
            # Translator returns None for symbol-only / empty input
            translated_text = result if result is not None else raw_text
            was_translated  = translated_text != raw_text
        except Exception:
            translated_text = raw_text   # fall back to original if API fails

    # Guard against None before stripping
    cleaned_text = strip_emojis_and_symbols(translated_text or "")

    return cleaned_text, lang, was_translated


def get_comments_for_video(client, video_id, max_comments=800):
    """
    Fetch comments, translate non-English ones, strip noise,
    and reject junk. Fetches up to 3× the target from the API
    to compensate for junk/language drop-off.
    """
    raw_pool        = []
    next_page_token = None
    fetch_target    = min(max_comments * 3, 3000)

    # ── Phase 1: collect raw comments from API ────────────────────────────
    while len(raw_pool) < fetch_target:

        batch = min(100, fetch_target - len(raw_pool))

        try:
            resp = client.commentThreads().list(
                videoId     = video_id,
                part        = "snippet",
                maxResults  = batch,
                textFormat  = "plainText",
                pageToken   = next_page_token,
                order       = "time",       # "time" gives more unique comments than "relevance"
            ).execute()

        except Exception as e:
            print(f"      API error: {e}")
            break

        for thread in resp.get("items", []):
            top      = thread["snippet"]["topLevelComment"]["snippet"]
            raw_text = top.get("textDisplay", "").strip()
            if not raw_text:
                continue
            raw_pool.append({
                "author":      top.get("authorDisplayName", ""),
                "rawText":     raw_text,
                "publishedAt": top.get("publishedAt", ""),
                "likeCount":   top.get("likeCount", 0),
            })

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    # ── Phase 2: clean, translate, filter ────────────────────────────────
    comments = []

    for c in raw_pool:

        if len(comments) >= max_comments:
            break

        cleaned_text, lang, was_translated = process_comment(c["rawText"])

        # Drop junk after cleaning
        if is_junk_comment(cleaned_text):
            continue

        comments.append({
            "author":         c["author"],
            "text":           cleaned_text,          # cleaned + translated
            "originalText":   c["rawText"],           # raw as received from API
            "originalLang":   lang,
            "translated":     was_translated,
            "publishedAt":    c["publishedAt"],
            "likeCount":      c["likeCount"],
        })

    return comments


# ── Parallel comment fetching ─────────────────────────────────────────────────

def fetch_comments_worker(video, max_comments):

    client = youtubeClient()

    try:
        comments = get_comments_for_video(
            client,
            video["videoId"],
            max_comments
        )

        video["comments"] = comments

        translated = sum(1 for c in comments if c["translated"])

        print(
            f"  ✓ {video['title'][:65]}\n"
            f"    -> {len(comments)} comments | {translated} translated"
        )

    except Exception as e:
        print(f"  ✗ {video['title'][:65]} -> Error: {e}")
        video["comments"] = []

    return video


def fetch_comments_parallel(videos, max_comments=800, max_workers=4):

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(fetch_comments_worker, v, max_comments)
            for v in videos
        ]

        for f in as_completed(futures):
            results.append(f.result())

    return results


# ── Main orchestrator ─────────────────────────────────────────────────────────

CANDIDATES_CACHE = "candidates_cache.json"


def fetchYoutubeData(
    searchQueries,
    maxVideosPerQuery  = 50,
    maxCommentsPerVideo= 800,
    maxFinalVideos     = 60,
    outputFile         = "youtubeDataDump.json",
    publishedAfter     = None,
    maxWorkers         = 4,
    minRelevanceScore  = 5,
):

    client = youtubeClient()

    # ── STEP 1: Collect candidate videos (with cache) ─────────────────────
    print("\n" + "=" * 65)
    print("STEP 1  Searching YouTube for Formula 1 safety videos")
    print("=" * 65)

    if os.path.exists(CANDIDATES_CACHE):
        print(f"  Loading candidates from cache: {CANDIDATES_CACHE}")
        with open(CANDIDATES_CACHE) as f:
            all_items = json.load(f)
        print(f"  {len(all_items)} candidates loaded (no API quota used)")

    else:
        all_items = []

        for query in searchQueries:
            print(f"  Query: '{query}'")
            try:
                items = search_videos_for_query(
                    client        = client,
                    query         = query,
                    max_videos    = maxVideosPerQuery,
                    published_after = publishedAfter,
                )
                print(f"    -> {len(items)} candidates found")
                all_items.extend(items)

            except HttpError as e:
                if "quotaExceeded" in str(e):
                    print(
                        f"  ⚠ Quota exceeded — stopping search early.\n"
                        f"    Proceeding with {len(all_items)} candidates collected so far."
                    )
                    break
                raise

        all_items = deduplicate(all_items)
        print(f"\nTotal unique candidates: {len(all_items)}")

        # Save so tomorrow's run skips search entirely
        with open(CANDIDATES_CACHE, "w") as f:
            json.dump(all_items, f)
        print(f"  Candidates cached to {CANDIDATES_CACHE}")

    # ── STEP 2: Relevance filtering ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2  Filtering and ranking by F1 safety relevance")
    print("=" * 65)

    filtered = rank_and_filter(
        all_items,
        min_score  = minRelevanceScore,
        max_videos = maxFinalVideos,
    )

    print(f"Videos passing filter: {len(filtered)}")

    if not filtered:
        print("No relevant videos found. Exiting.")
        return

    print("\nSelected videos:")
    for item in filtered:
        s     = item["snippet"]
        score = calculate_relevance_score(
            s.get("title", ""),
            s.get("description", ""),
            s.get("channelTitle", "")
        )
        print(
            f"  [{score:>3}] "
            f"{s.get('title','')[:65]}  |  "
            f"{s.get('channelTitle','')}"
        )

    # ── STEP 3: Metadata collection ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3  Fetching video metadata")
    print("=" * 65)

    video_ids = [item["id"]["videoId"] for item in filtered]
    videos    = get_video_details(client, video_ids)
    videos.sort(key=lambda v: v["viewCount"], reverse=True)

    print(f"Videos after metadata filter: {len(videos)}")

    # ── STEP 4: Comment collection ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4  Fetching, translating, and cleaning comments")
    print(
        f"        Target: "
        f"{maxCommentsPerVideo} comments × "
        f"{len(videos)} videos"
    )
    print("=" * 65)

    videos = fetch_comments_parallel(
        videos,
        max_comments = maxCommentsPerVideo,
        max_workers  = maxWorkers,
    )

    # ── STEP 5: Save JSON ─────────────────────────────────────────────────
    total_comments   = sum(len(v["comments"]) for v in videos)
    total_translated = sum(
        sum(1 for c in v["comments"] if c["translated"])
        for v in videos
    )

    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump({"videos": videos}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)
    print(f"  Videos saved   : {len(videos)}")
    print(f"  Total comments : {total_comments:,}")
    print(f"  Translated     : {total_translated:,}")
    print(f"  Output file    : {outputFile}")
    print("=" * 65)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    fetchYoutubeData(
        searchQueries      = SEARCH_QUERIES,
        maxVideosPerQuery  = 50,
        maxCommentsPerVideo= 800,
        maxFinalVideos     = 60,
        outputFile         = "youtubeDataDump.json",
        publishedAfter     = "2018-01-01T00:00:00Z",
        maxWorkers         = 4,
        minRelevanceScore  = 5,
    )