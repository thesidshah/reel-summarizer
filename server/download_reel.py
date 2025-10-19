#!/usr/bin/env python3
"""Download an Instagram reel video given its public URL."""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

GRAPHQL_ENDPOINT = (
    "https://www.instagram.com/graphql/query/"
    "?doc_id=8845758582119845&variables={variables}"
)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
APP_ID = "936619743392459"

SHORTCODE_RE = re.compile(r"/reel/([A-Za-z0-9_-]+)/?")


class ReelDownloadError(RuntimeError):
    pass


def extract_shortcode(url: str) -> str:
    """Return the shortcode component from a reel URL."""
    match = SHORTCODE_RE.search(url)
    if not match:
        raise ReelDownloadError(
            "Could not determine reel shortcode from the provided URL."
        )
    return match.group(1)


def fetch_metadata(shortcode: str) -> dict:
    """Fetch GraphQL metadata for the reel."""
    variables = json.dumps({"shortcode": shortcode})
    encoded = urllib.parse.quote(variables, safe="")
    url = GRAPHQL_ENDPOINT.format(variables=encoded)

    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("X-IG-App-ID", APP_ID)

    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 200:
                raise ReelDownloadError(
                    f"Metadata request failed with status {resp.status}."
                )
            data = resp.read()
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        raise ReelDownloadError(
            f"Metadata request returned HTTP error {exc.code}."
        ) from exc
    except urllib.error.URLError as exc:  # type: ignore[attr-defined]
        raise ReelDownloadError("Failed to reach Instagram endpoint.") from exc

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ReelDownloadError("Unexpected response format from Instagram.") from exc

    try:
        return payload["data"]["xdt_shortcode_media"]
    except KeyError as exc:
        raise ReelDownloadError("Metadata response missing media information.") from exc


def download_video(url: str, output: Path) -> None:
    """Download the video from the provided CDN URL."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)

    try:
        with urllib.request.urlopen(req) as resp, output.open("wb") as fh:
            if resp.status != 200:
                raise ReelDownloadError(
                    f"Video download failed with status {resp.status}."
                )
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                fh.write(chunk)
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        raise ReelDownloadError(
            f"Video download returned HTTP error {exc.code}."
        ) from exc
    except urllib.error.URLError as exc:  # type: ignore[attr-defined]
        raise ReelDownloadError("Failed to reach Instagram CDN.") from exc



def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download an Instagram reel video given its URL."
    )
    parser.add_argument("url", help="Public Instagram reel URL")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (defaults to <shortcode>.mp4)",
    )

    args = parser.parse_args(argv)

    try:
        shortcode = extract_shortcode(args.url)
        metadata = fetch_metadata(shortcode)
        video_url = metadata.get("video_url")
        if not video_url:
            raise ReelDownloadError("Video URL not present in metadata response.")

        output_path = args.output or Path(f"{shortcode}.mp4")
        download_video(video_url, output_path)
        print(f"Saved reel to {output_path}")
        return 0

    except ReelDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
