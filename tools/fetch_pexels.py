#!/usr/bin/env python3
"""Descarrega 1 clip vertical do Pexels. A chave vem so de PEXELS_API_KEY.
Uso: python3 fetch_pexels.py "typing on laptop" saida.mp4 credits.txt"""
import json, os, sys, urllib.parse, urllib.request

UA = "tiktok-videos/1.0 (+https://github.com/Paxeko530/tiktok-videos)"
query, out, credits = sys.argv[1], sys.argv[2], sys.argv[3]
idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
key = os.environ["PEXELS_API_KEY"]
def req(url):
    r = urllib.request.Request(url, headers={"Authorization": key, "User-Agent": UA})
    return json.load(urllib.request.urlopen(r, timeout=30))
d = req("https://api.pexels.com/videos/search?query=%s&orientation=portrait&size=medium&per_page=15" % urllib.parse.quote(query))
vids = [v for v in d.get("videos", []) if v["duration"] >= 8] or d.get("videos", [])
if not vids:
    sys.exit("Sem resultados para: " + query)
v = vids[idx % len(vids)]
fs = [f for f in v["video_files"] if f.get("file_type") == "video/mp4" and f.get("width")]
fs = [f for f in fs if f["height"] >= f["width"]] or fs
fs.sort(key=lambda f: abs(f["height"] - 1920))
dl = urllib.request.Request(fs[0]["link"], headers={"User-Agent": UA})
with urllib.request.urlopen(dl, timeout=120) as resp, open(out, "wb") as fh:
    fh.write(resp.read())
open(credits, "w").write("%s - %s (%s) - %s" % (os.path.basename(out), v["user"]["name"], v["user"]["url"], v["url"]) + chr(10))
print("ok", out, v["url"])
