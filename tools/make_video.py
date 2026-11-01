#!/usr/bin/env python3
"""Cria um video vertical 1080x1920: voz Kokoro (bm_george) + legendas palavra a palavra + fundo animado.
Uso: python3 make_video.py spec.json out.mp4 [--bg clip.mp4] [--model PASTA_KOKORO]
spec.json: {"lines": ["Frase 1."], "cta": "Follow for more AI tips."}
O fundo e gerado por codigo; com --bg usa um clip (escurecido) por baixo das legendas.
"""
import argparse, json, os, subprocess, tempfile, wave
import numpy as np
import sherpa_onnx
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
if not os.path.exists(FONT):
    FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def tts(lines, model, sid=26):
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{model}/model.onnx", voices=f"{model}/voices.bin", tokens=f"{model}/tokens.txt",
            data_dir=f"{model}/espeak-ng-data", dict_dir=f"{model}/dict", lexicon=f"{model}/lexicon-us-en.txt"),
        num_threads=2))
    t = sherpa_onnx.OfflineTts(cfg)
    chunks, sr = [], None
    for l in lines:
        a = t.generate(l, sid=sid, speed=1.0)
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
    chunks, sr = tts(lines, model)
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
    font = ImageFont.truetype(FONT, 92); small = ImageFont.truetype(FONT, 44)
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
    if bg:
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", bg, "-vf",
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},eq=brightness=-0.25",
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
            base = np.stack([18 + 30 * a, 20 + 45 * a, 48 + 90 * a], -1).astype(np.uint8)
        img = Image.fromarray(base); d = ImageDraw.Draw(img)
        d.text((W // 2, 150), "AI TIPS", font=small, fill=(255, 255, 255), anchor="mm")
        cur = next((p for p in prepared if p[0] - 0.05 <= t <= p[1] + 0.3), None)
        if cur:
            s, e, ws, rows, acc = cur
            lh = 120; y0 = H // 2 - lh * len(rows) // 2
            for r, row in enumerate(rows):
                full = " ".join(w for _, w in row); x = (W - d.textlength(full, font=font)) / 2
                for idx, w in row:
                    on = t >= acc[idx][0]
                    now = acc[idx][0] <= t < acc[idx][1]
                    col = (255, 214, 10) if now else ((255, 255, 255) if on else (120, 130, 160))
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
    p.add_argument("spec"); p.add_argument("out"); p.add_argument("--bg")
    p.add_argument("--model", default=os.path.expanduser("~/work/kokoro-multi-lang-v1_0"))
    a = p.parse_args()
    build(json.load(open(a.spec)), a.model, a.out, a.bg)
