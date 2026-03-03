import os, json, tempfile, urllib.request, urllib.parse

TEMP_DIR = os.path.join(tempfile.gettempdir(), "deepfake_tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

SOCIAL = ("youtube.com", "youtu.be", "tiktok.com", "facebook.com",
          "fb.watch", "instagram.com", "twitter.com", "x.com", "vimeo.com")

def download_video(url: str) -> str:
    try:
        is_social = any(d in url for d in SOCIAL)

        for f in os.listdir(TEMP_DIR):
            if f.startswith("video_tmp."):
                os.remove(os.path.join(TEMP_DIR, f))

        if is_social:
            try:
                import yt_dlp
            except ImportError:
                return json.dumps({"error": "yt-dlp not installed"})

            tmpl = os.path.join(TEMP_DIR, "video_tmp.%(ext)s")
            ydl_opts = {
                "format": "best[ext=mp4][height<=480]/best[height<=480]/best",
                "outtmpl": tmpl,
                "quiet": True,
                "no_warnings": True,
                "overwrites": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info  = ydl.extract_info(url, download=True)
                ext   = info.get("ext", "mp4")
                title = info.get("title", "video")[:40]

            dest = os.path.join(TEMP_DIR, f"video_tmp.{ext}")
            if not os.path.exists(dest):
                candidates = [os.path.join(TEMP_DIR, f)
                              for f in os.listdir(TEMP_DIR)
                              if f.startswith("video_tmp.")]
                if not candidates:
                    return json.dumps({"error": "yt-dlp did not save file"})
                dest = max(candidates, key=os.path.getmtime)

            return json.dumps({"path": dest, "name": f"{title}.{ext}"})
        else:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Linux; Android 11)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                ct = resp.headers.get("Content-Type", "")
                if "text/html" in ct:
                    return json.dumps({"error": "URL returns HTML. Use direct link."})
                data = resp.read()

            if len(data) < 10_000:
                return json.dumps({"error": f"File too small ({len(data)} bytes)"})

            fname = os.path.basename(urllib.parse.urlparse(url).path) or "video.mp4"
            dest  = os.path.join(TEMP_DIR, fname)
            with open(dest, "wb") as f:
                f.write(data)
            return json.dumps({"path": dest, "name": fname})

    except Exception as e:
        return json.dumps({"error": str(e)})