#!/bin/bash
#
# Search Spotify for tracks
#
# Usage:
#   ./spotify-search.sh "search query"
#
# The script reads SPOTIFY_TOKEN from .env.local in the project root,
# or you can set it directly: SPOTIFY_TOKEN="your_token" ./spotify-search.sh "query"
#
# Get a token from: https://developer.spotify.com/documentation/web-api/concepts/access-token

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load from .env.local if SPOTIFY_TOKEN not already set
if [ -z "$SPOTIFY_TOKEN" ] && [ -f "$PROJECT_ROOT/.env.local" ]; then
  SPOTIFY_TOKEN=$(grep "^SPOTIFY_TOKEN=" "$PROJECT_ROOT/.env.local" | cut -d '=' -f2-)
fi

if [ -z "$SPOTIFY_TOKEN" ]; then
  echo "Error: SPOTIFY_TOKEN not found"
  echo ""
  echo "Either set SPOTIFY_TOKEN in .env.local or pass it directly:"
  echo "  SPOTIFY_TOKEN=\"your_token\" $0 \"search query\""
  exit 1
fi

if [ -z "$1" ]; then
  echo "Error: Search query required"
  echo ""
  echo "Usage: SPOTIFY_TOKEN=\"your_token\" $0 \"search query\""
  exit 1
fi

QUERY=$(echo "$1" | sed 's/ /+/g')
curl -s "https://api.spotify.com/v1/search?q=${QUERY}&type=track&limit=5" \
  -H "Authorization: Bearer ${SPOTIFY_TOKEN}" | jq '.tracks.items[] | {name, album: .album.name, artists: [.artists[].name], id, uri, release_date: .album.release_date}'
