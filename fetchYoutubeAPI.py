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
# from 2020 onwards for sentiment analysis, topic modelling,
# and network analysis.
#

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator

from youtubeClient import youtubeClient

DetectorFactory.seed = 0


# Search queries — Formula 1 safety discussions from 2020 onwards

SEARCH_QUERIES = [
    "Formula 1 crash analysis",
    "F1 major accidents analysis",
    "Formula 1 safety discussion",
    "F1 halo safety analysis",
    "Formula 1 dangerous crashes",
    "Jules Bianchi crash analysis",
    "Romain Grosjean Bahrain crash analysis",
    "F1 crash reaction",
    "Formula 1 driver safety",
    "F1 FIA safety regulations",
    "Formula 1 accidents documentary",
    "F1 safety improvements",
    "Formula 1 controversial crashes",
    "F1 race accidents analysis",
    "Formula 1 crash aftermath discussion",
]


# Keyword filter lists

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


# Relevance scoring

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

    # Must mention Formula 1
    if contains_any(combined, FORMULA1_KEYWORDS):
        score += 4
    else:
        return -100

    # Reject unwanted content
    if contains_any(combined, EXCLUDE_KEYWORDS):
        return -100

    # Must contain F1 context
    if not contains_any(combined, F1_REQUIRED_KEYWORDS):
        return -100

    # Reward safety discussion
    if contains_any(combined, SAFETY_KEYWORDS):
        score += 5

    # Reward accident discussion
    if contains_any(combined, ACCIDENT_KEYWORDS):
        score += 4

    # Reward analytical discussion
    if contains_any(combined, ANALYSIS_KEYWORDS):
        score += 4

    # Reward contextual relevance
    hits = sum(1 for kw in STRONG_CONTEXT_KEYWORDS if kw in combined)
    score += min(hits, 5)

    # Reward official/reliable channels
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


# ISO 8601 duration parser

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


# YouTube API helpers

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


def rank_and_filter(items, min_score=7, max_videos=30):

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

        # Remove very short videos
        if duration < 180:
            continue

        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        title = snippet.get("title", "")
        desc = snippet.get("description", "")

        videos.append({
            "title": title,
            "videoId": item["id"],
            "channelTitle": snippet.get("channelTitle", ""),
            "publishedAt": snippet.get("publishedAt", ""),
            "viewCount": int(stats.get("viewCount", 0)),
            "likeCount": int(stats.get("likeCount", 0)),
            "commentCount": int(stats.get("commentCount", 0)),
            "durationSecs": duration,
            "category": assign_video_category(title, desc),
            "comments": [],
        })

    return videos


def get_comments_for_video(client, video_id, max_comments=500):

    comments = []
    next_page_token = None

    while len(comments) < max_comments:

        batch = min(100, max_comments - len(comments))

        try:

            resp = client.commentThreads().list(
                videoId=video_id,
                part="snippet",
                maxResults=batch,
                textFormat="plainText",
                pageToken=next_page_token,
                order="relevance",
            ).execute()

        except Exception as e:
            print(f"      API error: {e}")
            break

        for thread in resp.get("items", []):

            top = thread["snippet"]["topLevelComment"]["snippet"]

            raw_text = top.get("textDisplay", "")

            translated, lang = process_comment(raw_text)

            comments.append({
                "author": top.get("authorDisplayName", ""),
                "text": translated,
                "originalText": raw_text,
                "originalLang": lang,
                "publishedAt": top.get("publishedAt", ""),
                "likeCount": top.get("likeCount", 0),
            })

            if len(comments) >= max_comments:
                break

        next_page_token = resp.get("nextPageToken")

        if not next_page_token:
            break

    return comments


# Parallel comment fetching

def fetch_comments_worker(video, max_comments):

    client = youtubeClient()

    try:

        comments = get_comments_for_video(
            client,
            video["videoId"],
            max_comments
        )

        video["comments"] = comments

        translated = sum(
            1 for c in comments
            if c["originalLang"] != "en"
        )

        print(
            f"  ✓ {video['title'][:65]}\n"
            f"    -> {len(comments)} comments | {translated} translated"
        )

    except Exception as e:

        print(f"  ✗ {video['title'][:65]} -> Error: {e}")

        video["comments"] = []

    return video


def fetch_comments_parallel(
    videos,
    max_comments=500,
    max_workers=4
):

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(
                fetch_comments_worker,
                v,
                max_comments
            )
            for v in videos
        ]

        for f in as_completed(futures):
            results.append(f.result())

    return results


# Main orchestrator

def fetchYoutubeData(
    searchQueries,
    maxVideosPerQuery=50,
    maxCommentsPerVideo=500,
    maxFinalVideos=30,
    outputFile="youtubeDataDump.json",
    publishedAfter=None,
    maxWorkers=4,
    minRelevanceScore=7,
):

    client = youtubeClient()

    # STEP 1 — Collect candidate videos

    print("\n" + "=" * 65)
    print("STEP 1  Searching YouTube for Formula 1 safety videos")
    print("=" * 65)

    all_items = []

    for query in searchQueries:

        print(f"  Query: '{query}'")

        items = search_videos_for_query(
            client=client,
            query=query,
            max_videos=maxVideosPerQuery,
            published_after=publishedAfter,
        )

        print(f"    -> {len(items)} candidates found")

        all_items.extend(items)

    all_items = deduplicate(all_items)

    print(f"\nTotal unique candidates: {len(all_items)}")

    # STEP 2 — Relevance filtering

    print("\n" + "=" * 65)
    print("STEP 2  Filtering and ranking by F1 safety relevance")
    print("=" * 65)

    filtered = rank_and_filter(
        all_items,
        min_score=minRelevanceScore,
        max_videos=maxFinalVideos
    )

    print(f"Videos passing filter: {len(filtered)}")

    if not filtered:
        print("No relevant videos found. Exiting.")
        return

    print("\nSelected videos:")

    for item in filtered:

        s = item["snippet"]

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

    # STEP 3 — Metadata collection

    print("\n" + "=" * 65)
    print("STEP 3  Fetching video metadata")
    print("=" * 65)

    video_ids = [item["id"]["videoId"] for item in filtered]

    videos = get_video_details(client, video_ids)

    videos.sort(
        key=lambda v: v["viewCount"],
        reverse=True
    )

    print(f"Videos after filtering: {len(videos)}")

    # STEP 4 — Comment collection

    print("\n" + "=" * 65)
    print("STEP 4  Fetching and translating comments")
    print(
        f"        Target: "
        f"{maxCommentsPerVideo} comments × "
        f"{len(videos)} videos"
    )
    print("=" * 65)

    videos = fetch_comments_parallel(
        videos,
        max_comments=maxCommentsPerVideo,
        max_workers=maxWorkers
    )

    # STEP 5 — Save JSON

    total_comments = sum(
        len(v["comments"]) for v in videos
    )

    total_translated = sum(
        sum(
            1 for c in v["comments"]
            if c["originalLang"] != "en"
        )
        for v in videos
    )

    with open(outputFile, "w", encoding="utf-8") as f:

        json.dump(
            {"videos": videos},
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)
    print(f"  Videos saved   : {len(videos)}")
    print(f"  Total comments : {total_comments:,}")
    print(f"  Translated     : {total_translated:,}")
    print(f"  Output file    : {outputFile}")
    print("=" * 65)


# Entry point

if __name__ == "__main__":

    fetchYoutubeData(
        searchQueries=SEARCH_QUERIES,
        maxVideosPerQuery=60,
        maxCommentsPerVideo=500,
        maxFinalVideos=60,
        outputFile="youtubeDataDump.json",

        # Collect videos from 2020 onwards, including 2026/current videos
        publishedAfter="2020-01-01T00:00:00Z",

        maxWorkers=4,
        minRelevanceScore=6,
    )