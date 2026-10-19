"""
Resumable ingestion: playlist -> audio -> transcript -> chunks -> embeddings -> Qdrant

Two phases (run both, or one at a time):
  transcribe : download audio + run Whisper, save transcripts/<id>.json   (slow, GPU, needs internet only for download)
  upload     : chunk + embed + upsert to Qdrant                           (fast, needs internet)

Examples:
  python ingest.py "<playlist url>" --limit 3            # quick test on 3 videos
  python ingest.py "<playlist url>" --limit 20           # do 20 more videos, then stop
  python ingest.py "<playlist url>"                      # everything
  python ingest.py "<playlist url>" --phase transcribe   # only the long GPU phase
  python ingest.py --phase upload                        # only upload (no playlist URL needed)

Safe to stop (Ctrl+C, power cut, wifi drop) and re-run the same command at any time.
  - transcripts are written atomically, so they are never half-written
  - audio files are validated (via ffprobe) before use; broken ones are re-downloaded
  - network calls are retried with backoff; a video that still fails is skipped and reported
  - a video is only marked done (done/<id>.json) after ALL its chunks are confirmed in Qdrant
"""

import os, sys, site
if sys.platform == "win32":
    for sp in site.getsitepackages():
        for sub in ("cublas", "cudnn"):
            p = os.path.join(sp, "nvidia", sub, "bin")
            if os.path.isdir(p):
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

import argparse
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import yt_dlp
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PayloadSchemaType, PointStruct, VectorParams,
)

from common import COLLECTION, EMBEDDING_SIZE, get_client, get_embedder

AUDIO_DIR = Path("audio")
TRANSCRIPT_DIR = Path("transcripts")
DONE_DIR = Path("done")
for d in (AUDIO_DIR, TRANSCRIPT_DIR, DONE_DIR):
    d.mkdir(exist_ok=True)

CHUNK_SECONDS = 45
OVERLAP_SECONDS = 10
UPSERT_BATCH = 128

COOKIE_OPTS = {}
# ============================================================
# Helpers: retries and atomic file IO
# ============================================================

def with_retry(fn, *args, tries=8, what="network call", **kwargs):
    """Call fn; on failure wait 2,4,8,... seconds (max 60) and try again."""
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if "not a bot" in str(e):
                raise
            if attempt == tries:
                raise
            wait = min(2 ** attempt, 60)
            print(f"  ! {what} failed ({type(e).__name__}: {str(e)[:120]}). "
                  f"Retry {attempt}/{tries - 1} in {wait}s...")
            time.sleep(wait)


def atomic_write_json(path, data):
    """Write to a temp file, flush to disk, then rename. A power cut can never leave a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clean_stale_temp_files():
    for p in TRANSCRIPT_DIR.glob("*.tmp"):
        p.unlink(missing_ok=True)
    for p in DONE_DIR.glob("*.tmp"):
        p.unlink(missing_ok=True)


# ============================================================
# Playlist + audio
# ============================================================

def get_playlist_videos(playlist_url):
    opts = {"extract_flat": True, "quiet": True, "skip_download": True}

    def _fetch():
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(playlist_url, download=False)

    info = with_retry(_fetch, what="playlist fetch")
    return [
        {"id": e["id"], "title": e.get("title") or e["id"], "duration": e.get("duration")}
        for e in info["entries"] if e
    ]


def audio_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def audio_is_valid(path, expected_duration):
    """Reject missing, tiny, unreadable, or truncated audio files."""
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    if shutil.which("ffprobe") is None:
        return True                      # can't check further; trust the size
    dur = audio_duration(path)
    if dur is None or dur <= 0:
        return False
    if expected_duration and dur < expected_duration * 0.97 - 2:
        return False                     # file is shorter than the video: truncated
    return True


def download_audio(video):
    vid = video["id"]
    path = AUDIO_DIR / f"{vid}.mp3"

    if audio_is_valid(path, video.get("duration")):
        return path
    path.unlink(missing_ok=True)         # broken leftover from a previous interrupted run

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / f"{vid}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "sleep_interval": 5,
        "max_sleep_interval": 15,
        **COOKIE_OPTS,             # be gentle with YouTube to avoid throttling
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",    # plenty for speech
        }],
    }

    def _download():
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])

    with_retry(_download, tries=5, what="audio download")

    if not audio_is_valid(path, video.get("duration")):
        path.unlink(missing_ok=True)
        raise RuntimeError("audio file is incomplete or corrupted after download")
    return path


# ============================================================
# Transcription
# ============================================================

class Transcriber:
    def __init__(self, model_size, use_cpu, batch_size):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        if use_cpu:
            print(f"Loading Whisper '{model_size}' on CPU (slow)...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.batched = None
        else:
            print(f"Loading Whisper '{model_size}' on GPU...")
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            self.batched = BatchedInferencePipeline(self.model)
        self.batch_size = batch_size

    def run(self, audio_path):
        if self.batched is not None:
            segments, _ = self.batched.transcribe(str(audio_path), batch_size=self.batch_size)
        else:
            segments, _ = self.model.transcribe(str(audio_path), vad_filter=True)
        return [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments if s.text.strip()
        ]


def phase_transcribe(videos, args):
    print("\n=== PHASE 1: download + transcribe ===")
    transcriber = None
    processed, failed = 0, []

    for n, video in enumerate(videos, 1):
        if args.limit and processed >= args.limit:
            print(f"\nReached --limit {args.limit}; stopping this phase.")
            break

        vid, title = video["id"], video["title"]
        tpath = TRANSCRIPT_DIR / f"{vid}.json"

        if (DONE_DIR / f"{vid}.json").exists():
            continue                                    # already fully in Qdrant
        if tpath.exists():
            data = read_json(tpath)
            if data is not None and "segments" in data:
                continue                                # valid transcript already saved
            print(f"[{n}/{len(videos)}] {title}\n  corrupted transcript found, redoing")
            tpath.unlink(missing_ok=True)

        print(f"\n[{n}/{len(videos)}] {title}")
        started = time.time()
        try:
            audio = download_audio(video)
            if transcriber is None:
                transcriber = Transcriber(args.model, args.cpu, args.batch_size)
            segments = transcriber.run(audio)
            atomic_write_json(tpath, {"video_id": vid, "title": title, "segments": segments})
            audio.unlink(missing_ok=True)
            processed += 1
            print(f"  transcribed {len(segments)} segments in {(time.time() - started) / 60:.1f} min")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(title)
            if "not a bot" in str(e):
                print("YouTube is blocking downloads. Try again later or use cookies. Stopping.")
                break

    return failed


# ============================================================
# Chunking
# ============================================================

def chunk_segments(segments, window=CHUNK_SECONDS, overlap=OVERLAP_SECONDS):
    chunks = []
    n = len(segments)
    i = 0
    while i < n:
        start = segments[i]["start"]
        j = i
        while j < n and segments[j]["end"] - start <= window:
            j += 1
        if j == i:                       # one segment longer than the window
            j = i + 1

        group = segments[i:j]
        end = group[-1]["end"]
        chunks.append({
            "start": start,
            "end": end,
            "text": " ".join(s["text"] for s in group),
        })
        if j >= n:
            break

        k = i + 1                        # next chunk starts inside the overlap zone
        while k < j and segments[k]["start"] < end - overlap:
            k += 1
        i = k
    return chunks


# ============================================================
# Upload to Qdrant
# ============================================================

def setup_collection(client):
    if not with_retry(client.collection_exists, COLLECTION, what="Qdrant connect"):
        with_retry(
            client.create_collection,
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
            what="create collection",
        )
        with_retry(
            client.create_payload_index,
            collection_name=COLLECTION,
            field_name="video_id",
            field_schema=PayloadSchemaType.KEYWORD,
            what="create index",
        )


def mark_done(vid, title, n_chunks):
    atomic_write_json(DONE_DIR / f"{vid}.json", {"video_id": vid, "title": title, "chunks": n_chunks})


def phase_upload(args):
    print("\n=== PHASE 2: chunk + embed + upload ===")
    client = get_client()
    setup_collection(client)
    model = None
    processed, failed = 0, []

    for tfile in sorted(TRANSCRIPT_DIR.glob("*.json")):
        if args.limit and processed >= args.limit:
            print(f"\nReached --limit {args.limit}; stopping this phase.")
            break

        vid = tfile.stem
        if (DONE_DIR / f"{vid}.json").exists():
            continue

        data = read_json(tfile)
        if data is None or "segments" not in data:
            print(f"Skipping unreadable transcript {tfile.name} (re-run the transcribe phase)")
            tfile.unlink(missing_ok=True)
            continue

        title = data["title"]
        print(f"\n{title}")
        try:
            chunks = chunk_segments(data["segments"])
            if not chunks:
                print("  no speech found")
                mark_done(vid, title, 0)
                continue

            # If all chunks are already confirmed in Qdrant, just mark done.
            existing = with_retry(
                client.count,
                collection_name=COLLECTION,
                count_filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=vid))]),
                exact=True,
                what="count check",
            ).count
            if existing == len(chunks):
                print(f"  already in Qdrant ({existing} chunks)")
                mark_done(vid, title, existing)
                continue

            if model is None:
                print("Loading embedding model...")
                model = get_embedder()

            vectors = model.encode(
                [c["text"] for c in chunks], normalize_embeddings=True, batch_size=64
            )
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{vid}:{c['start']:.2f}")),
                    vector=vectors[i].tolist(),
                    payload={
                        "video_id": vid,
                        "title": title,
                        "start": round(c["start"], 2),
                        "end": round(c["end"], 2),
                        "text": c["text"],
                    },
                )
                for i, c in enumerate(chunks)
            ]

            for b in range(0, len(points), UPSERT_BATCH):
                with_retry(
                    client.upsert,
                    collection_name=COLLECTION,
                    points=points[b:b + UPSERT_BATCH],
                    what="upload",
                    tries=10,
                )

            # Verify everything arrived before declaring the video done.
            stored = with_retry(
                client.count,
                collection_name=COLLECTION,
                count_filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=vid))]),
                exact=True,
                what="verify",
            ).count
            if stored < len(chunks):
                raise RuntimeError(f"only {stored}/{len(chunks)} chunks stored")

            mark_done(vid, title, stored)
            processed += 1
            print(f"  stored {stored} chunks")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(title)

    return failed


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Index a YouTube playlist into Qdrant.")
    parser.add_argument("playlist_url", nargs="?", help="Playlist URL (not needed for --phase upload)")
    parser.add_argument("--phase", choices=["all", "transcribe", "upload"], default="all")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N new videos per phase, then stop (0 = no limit)")
    parser.add_argument("--model", default="small", help="Whisper model: tiny/base/small/medium")
    parser.add_argument("--batch-size", type=int, default=4, help="Whisper batch size (lower if GPU runs out of memory)")
    parser.add_argument("--cpu", action="store_true", help="Run Whisper on CPU instead of GPU")
    parser.add_argument("--cookies-from-browser", dest="browser", help="firefox, edge or chrome")   # NEW
    parser.add_argument("--cookies-file", help="path to a cookies.txt file")                        # NEW
    args = parser.parse_args()

    if args.browser:                                                                # NEW
        COOKIE_OPTS["cookiesfrombrowser"] = (args.browser,)                         # NEW
    if args.cookies_file:                                                           # NEW
        COOKIE_OPTS["cookiefile"] = args.cookies_file                               # NEW

    if args.phase in ("all", "transcribe") and not args.playlist_url:
        parser.error("playlist_url is required unless --phase upload")

    clean_stale_temp_files()
    failed = []

    try:
        if args.phase in ("all", "transcribe"):
            videos = get_playlist_videos(args.playlist_url)
            print(f"Found {len(videos)} videos in playlist")
            failed += phase_transcribe(videos, args)

        if args.phase in ("all", "upload"):
            failed += phase_upload(args)
    except KeyboardInterrupt:
        print("\n\nStopped by you. Progress is saved; re-run the same command to continue.")
        sys.exit(0)

    done = len(list(DONE_DIR.glob("*.json")))
    transcribed = len(list(TRANSCRIPT_DIR.glob("*.json")))
    print(f"\n--- Summary ---\nTranscripts saved: {transcribed}\nFully uploaded:    {done}")
    if failed:
        print(f"\n{len(failed)} video(s) failed this run:")
        for t in failed:
            print(f"  - {t}")
        print("Re-run the same command to retry them.")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()