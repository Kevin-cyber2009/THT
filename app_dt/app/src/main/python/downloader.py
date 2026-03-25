# downloader.py
# app/src/main/python/downloader.py

import json
import os
import re
import tempfile
from pathlib import Path

TEMP_DIR = os.path.join(tempfile.gettempdir(), "AIChecker_downloads")


def _cleanup():
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        for f in Path(TEMP_DIR).glob("video_tmp.*"):
            try: f.unlink()
            except: pass
    except: pass


def _normalize_url(url: str) -> str:
    url = url.strip()
    url = re.sub(r'/shortsy+/', '/shorts/', url)
    if url.startswith("/shorts/"):
        url = "https://www.youtube.com" + url
    elif url.startswith("/watch"):
        url = "https://www.youtube.com" + url
    elif url.startswith("/reel/"):
        url = "https://www.facebook.com" + url
    elif not url.startswith("http"):
        url = "https://" + url
    return url


def _is_social(url):
    return any(d in url.lower() for d in [
        "youtube.com", "youtu.be", "tiktok.com",
        "facebook.com", "fb.watch", "instagram.com",
        "twitter.com", "x.com", "vimeo.com",
    ])


def _find_file(title="video"):
    files = sorted(
        [f for f in Path(TEMP_DIR).glob("video_tmp.*") if f.stat().st_size > 5000],
        key=lambda f: f.stat().st_mtime, reverse=True
    )
    if files:
        f = files[0]
        safe = re.sub(r'[<>:"/\\|?*\n\r]', '_', str(title))[:50]
        return str(f), safe + f.suffix
    return None, None


def download_video(url: str) -> str:
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        _cleanup()
        url = _normalize_url(url)
        if _is_social(url):
            return _download_social(url)
        return _download_direct(url)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _try_ydl(yt_dlp, url, opts):
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info, None
    except Exception as e:
        return None, str(e)


def _download_social(url: str) -> str:
    try:
        import yt_dlp
    except ImportError:
        return json.dumps({"error": "yt-dlp chua cai dat"})

    out = os.path.join(TEMP_DIR, "video_tmp.%(ext)s")
    is_yt = "youtube.com" in url or "youtu.be" in url

    # Chuyen Shorts sang watch URL
    dl_url = url
    if "/shorts/" in url:
        vid = re.search(r'/shorts/([a-zA-Z0-9_-]+)', url)
        if vid:
            dl_url = f"https://www.youtube.com/watch?v={vid.group(1)}"

    # Base options - KHONG co format de override sau
    base = {
        "outtmpl": out,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "retries": 3,
        "fragment_retries": 3,
        "ignoreerrors": False,
        # Quan trong: merge format thanh 1 file
        "merge_output_format": "mp4",
    }

    # ===========================================================
    # DANH SACH CLIENT - thu tu tu it bi chan den nhieu bi chan
    # ios va android_vr la 2 client KHONG can PO token (nam 2024)
    # ===========================================================
    yt_strategies = [
        # --- iOS: it bi chan nhat, khong can PO token ---
        {
            "name": "ios",
            "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio/best",
            "extractor_args": {"youtube": {"player_client": ["ios"]}},
            "http_headers": {
                "User-Agent": "com.google.ios.youtube/19.29.1 (iPhone16,2; U; CPU iOS 17_5_1 like Mac OS X;)",
            },
        },
        # --- Android VR: client moi it bi chặn ---
        {
            "name": "android_vr",
            "format": "best[ext=mp4]/best",
            "extractor_args": {"youtube": {"player_client": ["android_vr"]}},
            "http_headers": {
                "User-Agent": "com.google.android.apps.youtube.vr.oculus/1.57.29 (Linux; U; Android 12L; eureka-user Build/SQ3A.220605.009.A1) gzip",
            },
        },
        # --- mweb: mobile web, nhe hon desktop web ---
        {
            "name": "mweb",
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "extractor_args": {"youtube": {"player_client": ["mweb"]}},
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
            },
        },
        # --- Android Creator Studio ---
        {
            "name": "android_creator",
            "format": "best[ext=mp4]/best",
            "extractor_args": {"youtube": {"player_client": ["android_creator"]}},
            "http_headers": {
                "User-Agent": "com.google.android.apps.youtube.creator/24.30.100 (Linux; U; Android 12; GB) gzip",
            },
        },
        # --- Android Music (co format rieng) ---
        {
            "name": "android_music",
            "format": "best[ext=mp4]/best",
            "extractor_args": {"youtube": {"player_client": ["android_music"]}},
            "http_headers": {
                "User-Agent": "com.google.android.apps.youtube.music/7.16.52 (Linux; U; Android 13; GB) gzip",
            },
        },
        # --- Android + iOS ket hop ---
        {
            "name": "android_ios",
            "format": "best",
            "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
            "http_headers": {
                "User-Agent": "com.google.android.youtube/19.02.39 (Linux; U; Android 12) gzip",
            },
        },
        # --- Fallback: khong chi dinh client ---
        {
            "name": "default",
            "format": "best[height<=480]/worst",
            "extractor_args": {},
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36",
            },
        },
    ]

    non_yt_strategies = [
        {
            "name": "best_mp4",
            "format": "best[ext=mp4]/best",
            "extractor_args": {},
            "http_headers": {"User-Agent": "Mozilla/5.0 (Android 12; Mobile)"},
        },
        {
            "name": "best",
            "format": "best",
            "extractor_args": {},
            "http_headers": {"User-Agent": "Mozilla/5.0 (Android 12; Mobile)"},
        },
    ]

    strategies = yt_strategies if is_yt else non_yt_strategies
    errors = []
    urls_to_try = list(dict.fromkeys([dl_url, url]))  # thu ca 2 URL

    for strat in strategies:
        for try_url in urls_to_try:
            _cleanup()
            opts = dict(base)
            opts["format"] = strat["format"]
            opts["extractor_args"] = strat.get("extractor_args", {})
            opts["http_headers"] = strat.get("http_headers", {})

            info, err = _try_ydl(yt_dlp, try_url, opts)
            if info:
                path, name = _find_file(info.get("title", "video"))
                if path:
                    return json.dumps({"path": path, "name": name})
            if err:
                errors.append(f"[{strat['name']}@{try_url[-20:]}]: {err[:60]}")
            break  # chi thu url thu nhat, neu that bai moi thu url thu 2

    # Thu tat ca URL voi tung strategy
    for strat in strategies:
        for try_url in urls_to_try[1:]:  # chi thu url thu 2
            _cleanup()
            opts = dict(base)
            opts["format"] = strat["format"]
            opts["extractor_args"] = strat.get("extractor_args", {})
            opts["http_headers"] = strat.get("http_headers", {})
            info, err = _try_ydl(yt_dlp, try_url, opts)
            if info:
                path, name = _find_file(info.get("title", "video"))
                if path:
                    return json.dumps({"path": path, "name": name})
            if err:
                errors.append(f"[{strat['name']}@URL2]: {err[:60]}")

    last = errors[-1] if errors else "Unknown"
    return json.dumps({
        "error": (
            f"YouTube dang chan tai video nay ({len(strategies)} client da thu).\n"
            f"Loi: {last}\n\n"
            "Cach don gian nhat:\n"
            "Dung app SNAPTUBE hoac 1DM tai video ve may,\n"
            "sau do mo AIChecker > SELECT FILE de phan tich."
        )
    })


def _download_direct(url: str) -> str:
    try:
        import urllib.request
        out = os.path.join(TEMP_DIR, "video_tmp.mp4")
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Android 12; Mobile)"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" in ctype:
                return json.dumps({"error": "Day la trang web, khong phai video."})
            data = resp.read()
        if len(data) < 10240:
            return json.dumps({"error": "File qua nho."})
        with open(out, "wb") as f:
            f.write(data)
        name = url.split("/")[-1].split("?")[0] or "video.mp4"
        if "." not in name:
            name += ".mp4"
        return json.dumps({"path": out, "name": name})
    except Exception as e:
        return json.dumps({"error": str(e)})