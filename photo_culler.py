#!/usr/bin/env python3
"""Photo Culler - desktop app (Windows / Mac / Linux).

Setup:   pip install pillow opencv-python numpy        (optional, for iPhone photos: pip install pillow-heif)
Run:     python photo_culler.py
Use:     Choose a folder -> review Picks / Similar shots / Rejects -> click photos to toggle -> Export.
Export copies your selected photos into a 'Picks' folder inside the one you chose. Originals are never changed.
"""
import math
import os
import shutil
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

EXT = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tif", ".tiff")
TARGET = {"Instagram 4:5": 0.8, "Reels / TikTok 9:16": 0.5625, "Square 1:1": 1.0}
WORK = 1600  # analysis resolution (long edge); sharpness is measured here, not on a tiny thumbnail


def clip(v, a=0.0, b=1.0):
    return max(a, min(b, v))


def analyze(path):
    im = Image.open(path)
    W, H = im.size
    if im.getexif().get(274) in (5, 6, 7, 8):
        W, H = H, W
    im.draft("RGB", (WORK, WORK))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((WORK, WORK))
    rgb = np.asarray(im)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape

    # Sharpness: light denoise first so grain/texture doesn't count as detail, then measure
    # per-block Laplacian variance. Blend a center-weighted average with the sharpest blocks,
    # so a razor-sharp background can't rescue a blurry subject and bokeh doesn't punish a sharp one.
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.8), cv2.CV_64F)
    n, v, cw = 8, [], []
    for by in range(n):
        for bx in range(n):
            v.append(lap[by * h // n:(by + 1) * h // n, bx * w // n:(bx + 1) * w // n].var())
            cw.append(math.exp(-(((bx - 3.5) / 3) ** 2 + ((by - 3.5) / 3) ** 2)))
    v, cw = np.array(v), np.array(cw)
    sharp = float(math.sqrt(np.average(v, weights=cw) * np.sort(v)[-16:].mean()))

    f = rgb.astype(np.float32)
    rg, yb = f[..., 0] - f[..., 1], 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2]
    color = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(int)
    th = im.copy()
    th.thumbnail((220, 220))
    return dict(path=path, name=os.path.basename(path), W=W, H=H, ratio=w / h, sharp=sharp,
                mean=float(g.mean()), std=float(g.std()), hi=float((g > 250).mean()),
                lo=float((g < 5).mean()), color=color,
                hash=(small[:, :-1] > small[:, 1:]).flatten(), thumb=th, sel=False)


def score(it, target, cutoff):
    sh = clip((math.log(max(it["sharp"], 1e-3)) - math.log(10)) / (math.log(90) - math.log(10)))
    ex = clip(1 - abs(it["mean"] - 120) / 110 - (it["hi"] + it["lo"]) * 2)
    co, cl = clip(it["std"] / 60), clip(it["color"] / 60)
    lowres = min(it["W"], it["H"]) < 1080
    ft = clip(1 - abs(math.log(it["ratio"] / target)) / math.log(2)) * (0.7 if lowres else 1)
    soft = it["sharp"] < cutoff
    bad_exp = it["mean"] < 60 or it["mean"] > 200 or it["hi"] > 0.08
    tags = []
    if soft: tags.append("soft / blurry")
    if it["mean"] < 60: tags.append("too dark")
    if bad_exp: tags.append("blown highlights" if it["mean"] >= 60 else "underexposed")
    if lowres: tags.append("low-res")
    if sh > 0.75: tags.append("sharp")
    if ft < 0.6: tags.append("needs crop")
    it["score"] = round(100 * (0.45 * sh + 0.2 * ex + 0.1 * co + 0.1 * cl + 0.15 * ft))
    it["rej"] = soft or it["mean"] < 60 or (bad_exp and ex < 0.4)
    it["tags"] = tags


class App:
    def __init__(s, root):
        s.root, s.items, s.tab, s.folder, s.imgs = root, [], "picks", None, []
        s.done, s.finished = 0, True
        root.title("Photo Culler")
        root.geometry("1120x780")
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Choose folder…", command=s.pick).pack(side="left")
        s.fmt = tk.StringVar(value="Instagram 4:5")
        cb = ttk.Combobox(top, textvariable=s.fmt, values=list(TARGET), state="readonly", width=20)
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: s.refresh())
        ttk.Label(top, text="Blur cutoff (higher = stricter)").pack(side="left")
        s.cut = tk.DoubleVar(value=20)
        sc = ttk.Scale(top, from_=8, to=60, variable=s.cut, length=140)
        sc.pack(side="left", padx=6)
        sc.bind("<ButtonRelease-1>", lambda e: s.refresh())
        ttk.Button(top, text="Export selected to 'Picks' folder", command=s.export).pack(side="right")
        s.status = ttk.Label(root, text="Choose a folder of photos to begin.", padding=(8, 0))
        s.status.pack(fill="x")
        s.bar = ttk.Progressbar(root)
        s.bar.pack(fill="x", padx=8, pady=4)
        tabs = ttk.Frame(root, padding=8)
        tabs.pack(fill="x")
        s.tb = {}
        for k in ("picks", "alt", "rej"):
            s.tb[k] = ttk.Button(tabs, command=lambda k=k: s.show(k))
            s.tb[k].pack(side="left", padx=2)
        wrap = ttk.Frame(root)
        wrap.pack(fill="both", expand=True)
        s.cv = tk.Canvas(wrap, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, command=s.cv.yview)
        s.cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        s.cv.pack(side="left", fill="both", expand=True)
        s.grid = ttk.Frame(s.cv)
        s.cv.create_window((0, 0), window=s.grid, anchor="nw")
        s.grid.bind("<Configure>", lambda e: s.cv.configure(scrollregion=s.cv.bbox("all")))
        s.cv.bind_all("<MouseWheel>", lambda e: s.cv.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units"))
        s.cv.bind_all("<Button-4>", lambda e: s.cv.yview_scroll(-1, "units"))
        s.cv.bind_all("<Button-5>", lambda e: s.cv.yview_scroll(1, "units"))
        s.label_tabs()

    def pick(s):
        d = filedialog.askdirectory(title="Choose a folder of photos")
        if not d:
            return
        files = [os.path.join(r, f) for r, _, fs in os.walk(d) if "Picks" not in r.split(os.sep)
                 for f in fs if f.lower().endswith(EXT)]
        if not files:
            messagebox.showinfo("Photo Culler", "No supported photos found in that folder.")
            return
        s.folder, s.items, s.done, s.finished = d, [], 0, False
        s.bar.configure(maximum=len(files), value=0)
        threading.Thread(target=s.work, args=(files,), daemon=True).start()
        s.poll(len(files))

    def work(s, files):
        def one(p):
            try:
                r = analyze(p)
            except Exception:
                r = None
            s.done += 1
            return r
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            res = list(ex.map(one, files))
        s.items = [r for r in res if r]
        s.finished = True

    def poll(s, n):
        s.bar["value"] = s.done
        if not s.finished:
            s.status.config(text=f"Analyzing {s.done} of {n}…")
            s.root.after(150, lambda: s.poll(n))
            return
        ham = lambda a, b: int(np.count_nonzero(a != b))
        gid = [-1] * len(s.items)
        k = 0
        for i, a in enumerate(s.items):
            if gid[i] < 0:
                gid[i] = k
                for j in range(i + 1, len(s.items)):
                    if gid[j] < 0 and ham(a["hash"], s.items[j]["hash"]) <= 10:
                        gid[j] = k
                k += 1
        for it, g in zip(s.items, gid):
            it["gid"] = g
        s.status.config(text=f"Analyzed {len(s.items)} photos. Click a photo to select or deselect it.")
        s.tab = "picks"
        s.refresh()

    def refresh(s):
        t = TARGET[s.fmt.get()]
        for it in s.items:
            score(it, t, s.cut.get())
        best = {}
        for it in s.items:
            if not it["rej"] and (it["gid"] not in best or it["score"] > best[it["gid"]]["score"]):
                best[it["gid"]] = it
        for it in s.items:
            it["cat"] = "rej" if it["rej"] else ("picks" if best[it["gid"]] is it else "alt")
            it["sel"] = it["cat"] == "picks"
        s.label_tabs()
        s.show(s.tab)

    def label_tabs(s):
        names = {"picks": "Picks", "alt": "Similar shots", "rej": "Rejects"}
        for k, b in s.tb.items():
            b.config(text=f"{names[k]} ({sum(1 for i in s.items if i.get('cat') == k)})")

    def show(s, k):
        s.tab = k
        for w in s.grid.winfo_children():
            w.destroy()
        s.imgs = []
        lst = sorted((i for i in s.items if i["cat"] == k), key=lambda i: -i["score"])
        for n, it in enumerate(lst):
            photo = ImageTk.PhotoImage(it["thumb"])
            s.imgs.append(photo)
            fr = tk.Frame(s.grid, highlightthickness=3, highlightbackground="#16a34a" if it["sel"] else "#cccccc")
            fr.grid(row=n // 5, column=n % 5, padx=5, pady=5, sticky="n")
            lb = tk.Label(fr, image=photo)
            lb.pack()
            nm = it["name"] if len(it["name"]) < 28 else it["name"][:25] + "…"
            tx = tk.Label(fr, text=f"{it['score']}  {nm}\n{', '.join(it['tags']) or ' '}", font=("TkDefaultFont", 9), width=30, wraplength=210)
            tx.pack()
            for wdg in (fr, lb, tx):
                wdg.bind("<Button-1>", lambda e, it=it, fr=fr: s.toggle(it, fr))
        s.cv.yview_moveto(0)

    def toggle(s, it, fr):
        it["sel"] = not it["sel"]
        fr.config(highlightbackground="#16a34a" if it["sel"] else "#cccccc")

    def export(s):
        chosen = [i for i in s.items if i.get("sel")]
        if not chosen:
            messagebox.showinfo("Photo Culler", "Nothing selected.")
            return
        dest = os.path.join(s.folder, "Picks")
        os.makedirs(dest, exist_ok=True)
        for it in chosen:
            name, target = it["name"], os.path.join(dest, it["name"])
            c = 1
            while os.path.exists(target):
                base, ext = os.path.splitext(name)
                target = os.path.join(dest, f"{base}_{c}{ext}")
                c += 1
            shutil.copy2(it["path"], target)
        messagebox.showinfo("Photo Culler", f"Copied {len(chosen)} photos to:\n{dest}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
