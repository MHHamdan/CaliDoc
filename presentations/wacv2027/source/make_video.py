"""Build the narrated talk videos from the deck PDF + video_scripts.js.

Narration: ElevenLabs text-to-speech with character timestamps (cached per text hash, so
re-runs cost nothing unless the script changes). Captions are the readable script, timed
from the alignment. Frames: the deck PDF rendered at 1920x1080. Output per variant:
../video/CaliDoc_WACV2027_Talk_<variant>.mp4 (+ soft caption track) and .srt.

Usage: ELEVENLABS_API_KEY=... python3 make_video.py [full|short ...]
Env:   FFMPEG (default: imageio-ffmpeg binary)
"""
import base64, hashlib, json, os, re, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "video"
CACHE = HERE / "video_cache"
PDF = HERE.parent / "CaliDoc_WACV2027_Presentation.pdf"
VOICE_ID = "onwK4e9ZLuTAKqWW03F9"  # ElevenLabs premade "Daniel - Steady Broadcaster"
MODEL = "eleven_multilingual_v2"
SETTINGS = {"stability": 0.6, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True, "speed": 1.05}
LEAD, TAIL, PAUSE = 0.5, 0.9, 0.7  # seconds
DISCLOSURE = "Narration: AI-generated voice (ElevenLabs) reading the authors' script"
FFMPEG = os.environ.get("FFMPEG") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
LIMITS = {"full": 600.0, "short": 300.0}

# Spoken forms for the synthesizer only; captions keep the written forms.
SAY = [
    (r"LayoutLMv3-Large", "Layout L M v3 Large"), (r"LayoutLMv3", "Layout L M v3"),
    (r"\bSROIE\b", "S-ROY"), (r"\bFUNSD\b", "F U N S D"), (r"WildReceipt", "Wild Receipt"),
    (r"\bCovGap\b", "Cov-Gap"), (r"SpanCRC", "Span C R C"), (r"\bBIO\b", "B I O"),
    (r"\bOCR\b", "O C R"), (r"\bF1\b", "F 1"), (r"CaliDoc", "Cali-Doc"), (r"N-SERC", "en-serk"), (r"\bHolm\b", "Hollm"), (r"ÉTS", "E T S"),
]


DIGITS = "zero one two three four five six seven eight nine".split()


def decimals(m):
    """1.000 -> 'one point zero zero zero', 0.0008 -> 'zero point zero zero zero eight' (no thousands misreads)."""
    whole = {"0": "zero", "1": "one"}.get(m.group(1), m.group(1))
    return whole + " point " + " ".join(DIGITS[int(d)] for d in m.group(2))


def spoken(s):
    for a, b in SAY:
        s = re.sub(a, b, s)
    return re.sub(r"\b(\d+)\.(\d+)\b", decimals, s)


def sentences(text):
    """Split narration into caption units: sentences, with [pause] as its own marker."""
    out = []
    for part in re.split(r"(\[pause\])", text):
        if part == "[pause]":
            out.append(None)
            continue
        out += [x.strip() for x in re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", part) if x.strip()]
    return out


def tts(text, prev_text, next_text):
    key = hashlib.sha256(json.dumps([VOICE_ID, MODEL, SETTINGS, text, prev_text, next_text]).encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    body = json.dumps({"text": text, "model_id": MODEL, "voice_settings": SETTINGS,
                       "previous_text": prev_text or None, "next_text": next_text or None}).encode()
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps?output_format=mp3_44100_128",
        data=body, headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"})
    try:
        res = json.loads(urllib.request.urlopen(req, timeout=300).read())
    except urllib.error.HTTPError as e:
        sys.exit(f"ElevenLabs HTTP {e.code}: {e.read().decode()[:400]}")
    f.write_text(json.dumps(res))
    return res


def ff(*args):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def duration(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        raise RuntimeError(f"no duration for {path}")
    h, m, s = m.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def frames(work):
    doc = fitz.open(PDF)
    paths = []
    for i, page in enumerate(doc, 1):
        p = work / f"frame{i:02d}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(1920 / page.rect.width, 1080 / page.rect.height))
        pix.save(p)
        paths.append(p)
    # disclosure on the title frame
    im = Image.open(paths[0]).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(str(Path.home() / ".fonts/calidoc-metric/Carlito-Regular.ttf"), 26)
    except OSError:
        font = ImageFont.load_default()
    w = d.textlength(DISCLOSURE, font=font)
    d.text((1920 - 86 - w, 1080 - 62), DISCLOSURE, fill=(203, 213, 225), font=font)
    im.save(paths[0])
    return paths


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def wrap_caption(s, width=42):
    """Split long sentences into <=2-line caption chunks."""
    words, chunks, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 2 * width and cur:
            chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    chunks.append(cur)
    out = []
    for c in chunks:  # break each chunk into two lines near the middle
        if len(c) > width:
            mid = c.rfind(" ", 0, len(c) // 2 + 10)
            c = c[:mid] + "\n" + c[mid + 1:]
        out.append(c)
    return out


def build(variant, scripts):
    work = CACHE / f"work_{variant}"
    work.mkdir(parents=True, exist_ok=True)
    imgs = frames(work)
    segs, cues, chapters, t0 = [], [], [], 0.0
    items = scripts[variant]
    for k, item in enumerate(items):
        units = sentences(item["text"])
        tts_text = " ".join('<break time="%.1fs" />' % PAUSE if u is None else spoken(u) for u in units)
        prev_t = spoken(items[k - 1]["text"].replace("[pause]", "")) if k else ""
        next_t = spoken(items[k + 1]["text"].replace("[pause]", "")) if k + 1 < len(items) else ""
        res = tts(tts_text, prev_t[-300:], next_t[:300])
        mp3 = work / f"s{k:02d}.mp3"
        mp3.write_bytes(base64.b64decode(res["audio_base64"]))
        al = res.get("alignment") or res.get("normalized_alignment")
        chars, starts, ends = "".join(al["characters"]), al["character_start_times_seconds"], al["character_end_times_seconds"]
        # caption timing: locate each spoken sentence in the aligned character stream
        pos = 0
        for u in units:
            if u is None:
                continue
            su = spoken(u)
            j = chars.find(su[:24], pos)
            if j < 0:
                j = pos
            e = min(len(ends) - 1, j + len(su) - 1)
            a, b = starts[min(j, len(starts) - 1)], ends[e]
            pieces = wrap_caption(u)
            n = sum(len(spoken(p)) for p in pieces)
            acc = a
            for p in pieces:
                dt = (b - a) * len(spoken(p)) / n
                cues.append((t0 + LEAD + acc, t0 + LEAD + acc + dt, p))
                acc += dt
            pos = j + len(su)
        wav = work / f"s{k:02d}.wav"
        ff("-i", str(mp3), "-af", f"adelay={int(LEAD*1000)}:all=1,apad=pad_dur={TAIL}", "-ar", "48000", "-ac", "2", str(wav))
        dur = duration(wav)
        seg = work / f"seg{k:02d}.mp4"
        ff("-loop", "1", "-framerate", "30", "-i", str(imgs[item["slide"] - 1]), "-i", str(wav), "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-tune", "stillimage", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-g", "300", "-c:a", "aac", "-b:a", "192k", "-shortest", str(seg))
        segs.append(seg)
        chapters.append((t0, item["slide"]))
        t0 += dur
        print(f"  {variant} slide {item['slide']:2d}: {dur:6.1f}s  (cum {t0/60:5.2f} min)", flush=True)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    joined = work / "joined.mp4"
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined))
    OUT.mkdir(exist_ok=True)
    srt = OUT / f"CaliDoc_WACV2027_Talk_{variant}.srt"
    srt.write_text("".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(cues, 1)))
    titles = {n["id"]: n["title"] for n in scripts["titles"]}
    (OUT / f"CaliDoc_WACV2027_Talk_{variant}.chapters.txt").write_text(
        "".join(f"{int(t // 60)}:{int(t % 60):02d} {titles[sl]}\n" for t, sl in chapters))
    mp4 = OUT / f"CaliDoc_WACV2027_Talk_{variant}.mp4"
    ff("-i", str(joined), "-i", str(srt), "-map", "0:v", "-map", "0:a", "-map", "1:s", "-c:v", "copy",
       "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
       "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-metadata", "title=CaliDoc (WACV 2027, paper 1597)",
       "-metadata", f"comment={DISCLOSURE}", "-movflags", "+faststart", str(mp4))
    total = duration(mp4)
    print(f"{variant}: {total/60:.2f} min, {mp4.stat().st_size/1e6:.1f} MB, {len(cues)} captions -> {mp4}")
    if total > LIMITS[variant]:
        print(f"WARNING: {variant} exceeds {LIMITS[variant]/60:.0f} min")


if __name__ == "__main__":
    scripts = json.loads(subprocess.check_output(["node", "-e", f"console.log(JSON.stringify(require('{HERE}/video_scripts.js')))"]))
    scripts["titles"] = json.loads(subprocess.check_output(["node", "-e", f"console.log(JSON.stringify(require('{HERE}/notes.js')))"]))
    for v in sys.argv[1:] or ["full", "short"]:
        build(v, scripts)
