"""Sprint pose tracker: video -> cleaned 13-joint CSV (same schema as before, plus visibility and interpolated).

Command line:  python sprint_tracker.py sprint_video1.mp4 [--near-side left|right] [--sigma 1.5]
Or import track_video() (tracker_server.py does this).
"""
import argparse
import csv

import cv2
import mediapipe as mp
import numpy as np

JOINT_NAMES = ["head", "neck", "elbow_front", "wrist_front", "elbow_rear", "wrist_rear", "pelvis",
               "front_knee", "front_heel", "front_toe", "rear_knee", "rear_heel", "rear_toe"]

# MediaPipe landmark ids used, and their column in the raw array
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_HE, R_HE, L_FT, R_FT = \
    0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30, 31, 32
LM = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_HE, R_HE, L_FT, R_FT]
IDX = {m: i for i, m in enumerate(LM)}

# (rows of near-side joints, rows of far-side joints, left landmarks, right landmarks)
GROUPS = [
    ([2, 3], [4, 5], (L_EL, L_WR), (R_EL, R_WR)),
    ([7, 8, 9], [10, 11, 12], (L_KN, L_HE, L_FT), (R_KN, R_HE, R_FT)),
]

VIS_CORE = 0.4    # shoulders and hips must be at least this visible for the athlete to count as "in frame"
VIS_POINT = 0.3   # a single leg/head point below this visibility is treated as missing
VIS_ARM = 0.1     # arms are often half hidden behind the torso, so they get a lower bar
MIN_BODY = 0.10   # nose-to-knee height must be at least this fraction of the frame height
MIN_RUN = 5       # in-frame stretches shorter than this are discarded (phantom detections)
MAX_GAP = 6       # longest gap (frames) that gets interpolated
ARM_ROWS = (2, 3, 4, 5)  # elbow/wrist rows in JOINT_NAMES
ARM_GAP = 15      # arms may be interpolated across longer gaps


def run_mediapipe(video_path, progress=None, complexity=2, t_start=0.0, t_end=None):
    """Raw landmarks, no smoothing: array (frames, 15, 4) of x, y, z, visibility (NaN when nothing detected)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:
        print("WARNING: could not read fps from the video, using 25.0")
        fps = 25.0
    first = int(t_start * fps)
    last = int(t_end * fps) if t_end else None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if last and total:
        total = min(total, last)
    total = max(0, total - first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=complexity, smooth_landmarks=False,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    rows, w, h = [], 0, 0
    while last is None or first + len(rows) < last:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        m = max(w, h)
        small = cv2.resize(frame, None, fx=1280 / m, fy=1280 / m) if m > 1280 else frame  # model doesn't need more
        res = pose.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        a = np.full((len(LM), 4), np.nan, np.float32)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            for k, i in enumerate(LM):
                a[k] = (lm[i].x * w, lm[i].y * h, lm[i].z * w, lm[i].visibility)
        rows.append(a)
        if progress and len(rows) % 10 == 0:
            progress(len(rows), total)
    cap.release()
    pose.close()
    if not rows:
        raise RuntimeError("The video has no readable frames in that time range")
    pad = np.full((first, len(LM), 4), np.nan, np.float32)  # skipped frames, so frame numbers still match the video
    return np.concatenate([pad, np.array(rows)]), fps, h


def presence_mask(raw, height):
    """True where a real, full-size athlete is detected. Returns (mask, body size in pixels per frame)."""
    core = [IDX[m] for m in (NOSE, L_SH, R_SH, L_HIP, R_HIP, L_KN, R_KN)]
    torso = [IDX[m] for m in (L_SH, R_SH, L_HIP, R_HIP)]
    vis = np.nan_to_num(raw[:, torso, 3], nan=0.0)
    ys = np.nan_to_num(raw[:, core, 1], nan=0.0)
    size = ys.max(axis=1) - ys.min(axis=1)
    return (vis.min(axis=1) > VIS_CORE) & (size > MIN_BODY * height), size


def find_runs(mask):
    """In-frame stretches: bridge short dropouts, then drop stretches shorter than MIN_RUN."""
    runs, s = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if runs and s - runs[-1][1] <= MAX_GAP:
                runs[-1][1] = i
            else:
                runs.append([s, i])
            s = None
    return [(a, b) for a, b in runs if mask[a:b].sum() >= MIN_RUN]


def pick_near_side(raw, run):
    """Side facing the camera, from hip depth over the whole longest in-frame stretch (smaller z = closer)."""
    seg = raw[run[0]:run[1]]
    d = np.nanmedian(seg[:, IDX[L_HIP], 2] - seg[:, IDX[R_HIP], 2])
    return "left" if d < 0 else "right"


def dist(a, b):
    return np.nansum(np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]))


def assign_limbs(raw, present, near, body):
    """Build the 13 joints (x, y, vis). Near/far limbs follow the locked side, but if a limb's points are much
    closer to the previous frame's far limb than its near limb (a left/right label swap when the legs cross),
    the two are swapped back. Returns (joints, number of frames re-labelled)."""
    n = len(raw)
    out = np.full((n, len(JOINT_NAMES), 3), np.nan, np.float32)
    fixed = 0
    nose = raw[:, IDX[NOSE]]
    out[:, 0] = nose[:, [0, 1, 3]]
    for row, (a, b) in ((1, (L_SH, R_SH)), (6, (L_HIP, R_HIP))):
        pa, pb = raw[:, IDX[a]], raw[:, IDX[b]]
        out[:, row, :2] = (pa[:, :2] + pb[:, :2]) / 2
        out[:, row, 2] = np.minimum(pa[:, 3], pb[:, 3])
    for near_rows, far_rows, lm_l, lm_r in GROUPS:
        prev = None
        for f in range(n):
            if not present[f]:
                prev = None
                continue
            l, r = raw[f, [IDX[m] for m in lm_l]], raw[f, [IDX[m] for m in lm_r]]
            nr, fr = (l, r) if near == "left" else (r, l)
            if prev is not None:
                keep = dist(nr, prev[0]) + dist(fr, prev[1])
                swap = dist(fr, prev[0]) + dist(nr, prev[1])
                if swap < 0.7 * keep and keep - swap > 0.1 * body * len(nr):
                    nr, fr = fr, nr
                    fixed += 1
            out[f, near_rows] = nr[:, [0, 1, 3]]
            out[f, far_rows] = fr[:, [0, 1, 3]]
            if prev is None:
                prev = [nr[:, :2].copy(), fr[:, :2].copy()]
            else:
                for p, c in ((prev[0], nr), (prev[1], fr)):
                    ok = ~np.isnan(c[:, 0])
                    p[ok] = c[ok, :2]
    return out, fixed


def clean(j, present):
    """Blank out frames with no athlete, low-visibility points, and impossible thigh/shank lengths (merged legs)."""
    j = j.copy()
    j[~present] = np.nan
    thr = np.full(j.shape[1], VIS_POINT)
    thr[list(ARM_ROWS)] = VIS_ARM  # keep hidden-arm estimates; they are flagged by their low visibility
    j[np.nan_to_num(j[..., 2], nan=0.0) < thr, :2] = np.nan
    for top, mid, end in ((6, 7, 8), (6, 10, 11), (1, 2, 3), (1, 4, 5)):
        for a, b in ((top, mid), (mid, end)):
            d = np.hypot(j[:, a, 0] - j[:, b, 0], j[:, a, 1] - j[:, b, 1])
            med = np.nanmedian(d)
            j[(d < 0.5 * med) | (d > 1.6 * med), b, :2] = np.nan
    return j


def gap_lengths(ok):
    g, i = np.zeros(len(ok), int), 0
    while i < len(ok):
        if ok[i]:
            i += 1
            continue
        k = i
        while k < len(ok) and not ok[k]:
            k += 1
        g[i:k] = k - i
        i = k
    return g


def smooth(y, sigma):
    """Zero-lag Gaussian smoothing of each unbroken stretch (symmetric kernel, so no delay on fast limbs)."""
    if sigma <= 0:
        return y
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out, ok, i = y.copy(), ~np.isnan(y), 0
    while i < len(y):
        if not ok[i]:
            i += 1
            continue
        e = i
        while e < len(y) and ok[e]:
            e += 1
        if e - i > 2:
            out[i:e] = np.convolve(np.pad(y[i:e], r, mode="edge"), k, mode="valid")
        i = e
    return out


def fill_and_smooth(j, runs, sigma):
    """Interpolate gaps of up to MAX_GAP frames inside each run, then smooth. Returns the 'interpolated' flags."""
    filled = np.zeros(j.shape[:2], bool)
    for a, b in runs:
        x = np.arange(b - a)
        for k in range(j.shape[1]):
            for c in (0, 1):
                y = j[a:b, k, c]
                ok = ~np.isnan(y)
                if ok.sum() < 2:
                    continue
                inside = (x > x[ok][0]) & (x < x[ok][-1])
                fill = ~ok & inside & (gap_lengths(ok) <= (ARM_GAP if k in ARM_ROWS else MAX_GAP))
                y2 = np.where(ok | fill, np.interp(x, x[ok], y[ok]), np.nan)
                filled[a:b, k] |= fill
                j[a:b, k, c] = smooth(y2, sigma)
    return filled


def write_csv(path, j, filled):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "joint_name", "joint_number", "x", "y", "visibility", "interpolated"])
        for f in range(len(j)):
            for k, name in enumerate(JOINT_NAMES):
                x, y, v = j[f, k]
                if np.isnan(x) or np.isnan(y):
                    w.writerow([f, name, k + 1, "", "", "" if np.isnan(v) else f"{v:.3f}", 0])
                else:
                    w.writerow([f, name, k + 1, f"{x:.2f}", f"{y:.2f}", "" if np.isnan(v) else f"{v:.3f}",
                                int(filled[f, k])])


def track_video(video_path, csv_path="sprint_coordinates.csv", fps_path="sprint_video_fps.txt",
                near_side=None, sigma=1.5, progress=None, complexity=2, t_start=0.0, t_end=None):
    raw, fps, height = run_mediapipe(video_path, progress, complexity, t_start, t_end)
    mask, size = presence_mask(raw, height)
    runs = find_runs(mask)
    if not runs:
        raise RuntimeError("No athlete detected. Check that the whole body is in frame and well lit.")
    present = np.zeros(len(raw), bool)
    for a, b in runs:
        present[a:b] = True
    raw[~present] = np.nan
    body = float(np.median(size[mask]))
    near = near_side or pick_near_side(raw, max(runs, key=lambda r: r[1] - r[0]))
    joints, fixed = assign_limbs(raw, present, near, body)
    joints = clean(joints, present)
    filled = fill_and_smooth(joints, runs, sigma)
    write_csv(csv_path, joints, filled)
    with open(fps_path, "w") as fh:
        fh.write(f"{fps:.6f}\n")
    return {"fps": fps, "frames": len(raw), "present_frames": int(present.sum()), "near_side": near,
            "relabelled_frames": fixed, "csv": csv_path, "first_frame": int(np.argmax(present)),
            "last_frame": int(len(present) - 1 - np.argmax(present[::-1]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--csv", default="sprint_coordinates.csv")
    ap.add_argument("--near-side", choices=["left", "right"], help="side facing the camera (default: auto)")
    ap.add_argument("--sigma", type=float, default=1.5, help="smoothing in frames, 0 = off")
    ap.add_argument("--model", type=int, choices=[0, 1, 2], default=2, help="2 = most accurate and slowest")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="end time in seconds")
    a = ap.parse_args()
    print(track_video(a.video, a.csv, near_side=a.near_side, sigma=a.sigma, complexity=a.model,
                      t_start=a.start, t_end=a.end,
                      progress=lambda i, n: print(f"\r{i}/{n} frames", end="")))