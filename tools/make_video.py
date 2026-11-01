#!/usr/bin/env python3
"""Cria um video vertical 1080x1920: voz Kokoro (bm_george) + legendas palavra a palavra + fundo animado.
Uso: python3 make_video.py spec.json out.mp4 [--bg clip.mp4] [--model PASTA_KOKORO]
spec.json: {"lines": ["Frase 1."], "cta": "Follow for more AI tips."}
Variacao opcional no spec: "style" (classic, mint, sunset, rose, mono, ice; sem isto escolhe-se um pelo nome do ficheiro),
"label" (texto do topo), "speed" (0.9-1.15), "voice" (sid Kokoro, por defeito 26 = bm_george), "caption_pos" (0.3-0.7), "bar" (true/false).
O fundo e gerado por codigo; com --bg usa um clip (escurecido) por baixo das legendas.
"""
import argparse, json, os, subprocess, tempfile, wave, zlib
import numpy as np
import sherpa_onnx
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
if not os.path.exists(FONT):
    FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

STYLES = {
    "classic": dict(hi=(255, 214, 10), on=(255, 255, 255), off=(120, 130, 160), pos=0.50, size=92, dim=-0.25, bar=False, pal=(18, 20, 48, 30, 45, 90)),
    "mint":    dict(hi=(0, 240, 170), on=(255, 255, 255), off=(110, 150, 140), pos=0.56, size=88, dim=-0.30, bar=True, pal=(8, 30, 28, 20, 70, 60)),
    "sunset":  dict(hi=(255, 140, 60), on=(255, 245, 235), off=(160, 120, 110), pos=0.44, size=96, dim=-0.30, bar=True, pal=(50, 18, 30, 110, 50, 40)),
    "rose":    dict(hi=(255, 105, 180), on=(255, 255, 255), off=(150, 120, 150), pos=0.52, size=90, dim=-0.28, bar=False, pal=(40, 14, 50, 90, 30, 100)),
    "mono":    dict(hi=(255, 255, 255), on=(235, 235, 235), off=(110, 110, 110), pos=0.60, size=100, dim=-0.38, bar=True, pal=(12, 12, 12, 40, 40, 40)),
    "ice":     dict(hi=(120, 220, 255), on=(255, 255, 255), off=(120, 140, 170), pos=0.46, size=90, dim=-0.28, bar=False, pal=(10, 24, 50, 30, 80, 130)),
}

def tts(lines, model, sid=26, speed=1.0):
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{model}/model.onnx", voices=f"{model}/voices.bin", tokens=f"{model}/tokens.txt",
            data_dir=f"{model}/espeak-ng-data", dict_dir=f"{model}/dict", lexicon=f"{model}/lexicon-us-en.txt"),
        num_threads=2))
    t = sherpa_onnx.OfflineTts(cfg)
    chunks, sr = [], None
    for l in lines:
        a = t.generate(l, sid=sid, speed=speed)
        sr = a.sample_rate
        chunks.append(np.array(a.samples, dtype=np.float32))
    return chunks, sr

def wrap(draw, words, font, maxw):
    rows, cur = [], []
    for i, w in enumerate(words):
        test = " ".join(x for _, x in cur + [(i, w)])
        if cur and draw.textlength(test, font=font) > maxw:
            rows.append(cur); cur = []
        cur.append((i, w))
    rows.append(cur)
    return rows

def build(spec, model, out, bg):
    lines = spec["lines"] + ([spec["cta"]] if spec.get("cta") else [])
    name = spec.get("style") or list(STYLES)[zlib.crc32(os.path.basename(out).encode()) % len(STYLES)]
    st = dict(STYLES[name]); st["pos"] = spec.get("caption_pos", st["pos"]); st["bar"] = spec.get("bar", st["bar"])
    label = spec.get("label", "AI TIPS")
    chunks, sr = tts(lines, model, sid=spec.get("voice", 26), speed=spec.get("speed", 1.0))
    print("style", name, st["pos"], "speed", spec.get("speed", 1.0))
    gap = int(0.35 * sr)
    audio, segs, pos = [], [], 0
    for c in chunks:
        segs.append((pos / sr, (pos + len(c)) / sr)); audio += [c, np.zeros(gap, np.float32)]; pos += len(c) + gap
    audio = np.concatenate(audio)
    total = len(audio) / sr + 0.6
    wav = os.path.join(tempfile.gettempdir(), os.path.basename(out) + ".wav")
    with wave.open(wav, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    font = ImageFont.truetype(FONT, st["size"]); small = ImageFont.truetype(FONT, 44)
    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    prepared = []
    for (s, e), l in zip(segs, lines):
        ws = l.split(); n = sum(len(w) for w in ws)
        t0, acc = s, []
        for w in ws:
            d = (e - s) * len(w) / n; acc.append((t0, t0 + d)); t0 += d
        prepared.append((s, e, ws, wrap(dummy, ws, font, W - 160), acc))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nframes = int(total * FPS)
    dec = None
    if bg and len(bg) > 1:
        seg = total / len(bg) + 0.1
        cat = os.path.join(tempfile.gettempdir(), os.path.basename(out) + ".bg.mp4")
        ins = []
        for b in bg: ins += ["-stream_loop", "-1", "-t", f"{seg:.2f}", "-i", b]
        fc = "".join(f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},setsar=1[v{i}];" for i in range(len(bg)))
        fc += "".join(f"[v{i}]" for i in range(len(bg))) + f"concat=n={len(bg)}:v=1:a=0[o]"
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex", fc, "-map", "[o]", "-an", cat], check=True)
        bg = [cat]
    if bg:
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", bg[0], "-vf",
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},eq=brightness={st['dim']}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", out], stdin=subprocess.PIPE)
    for i in range(nframes):
        t = i / FPS
        if dec:
            raw = dec.stdout.read(W * H * 3)
            base = np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy() if len(raw) == W * H * 3 else np.zeros((H, W, 3), np.uint8)
        else:
            a = 0.5 + 0.5 * np.sin(yy / 500 + t * 0.6) * np.cos(xx / 400 - t * 0.4)
            p = st["pal"]
            base = np.stack([p[0] + p[3] * a, p[1] + p[4] * a, p[2] + p[5] * a], -1).astype(np.uint8)
        img = Image.fromarray(base); d = ImageDraw.Draw(img)
        if label:
            d.text((W // 2, 150), label, font=small, fill=(255, 255, 255), anchor="mm")
        if st["bar"]:
            d.rectangle((0, H - 14, int(W * min(1, t / total)), H), fill=st["hi"])
        cur = next((p for p in prepared if p[0] - 0.05 <= t <= p[1] + 0.3), None)
        if cur:
            s, e, ws, rows, acc = cur
            lh = 120; y0 = int(H * st["pos"]) - lh * len(rows) // 2
            for r, row in enumerate(rows):
                full = " ".join(w for _, w in row); x = (W - d.textlength(full, font=font)) / 2
                for idx, w in row:
                    on = t >= acc[idx][0]
                    now = acc[idx][0] <= t < acc[idx][1]
                    col = st["hi"] if now else (st["on"] if on else st["off"])
                    d.text((x + 3, y0 + r * lh + 3), w, font=font, fill=(0, 0, 0))
                    d.text((x, y0 + r * lh), w, font=font, fill=col)
                    x += d.textlength(w + " ", font=font)
        enc.stdin.write(img.tobytes())
    enc.stdin.close(); enc.wait()
    if dec: dec.kill()
    os.remove(wav)
    print(f"ok {out} {total:.1f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("spec"); p.add_argument("out"); p.add_argument("--bg", nargs="*")
    p.add_argument("--model", default=os.path.expanduser("~/work/kokoro-multi-lang-v1_0"))
    a = p.parse_args()
    build(json.load(open(a.spec)), a.model, a.out, a.bg)
