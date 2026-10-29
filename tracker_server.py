"""Local server: serves the analyzer page and runs sprint_tracker on the video you pick in it.

    pip install flask
    python tracker_server.py        then open  http://localhost:5000
Keep this file, sprint_tracker.py and sprint_frame_analyzer.html in the same folder.
"""
import os
import tempfile
import threading
import uuid

from flask import Flask, jsonify, request, send_file, send_from_directory

from sprint_tracker import track_video

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = tempfile.mkdtemp(prefix="sprint_")
jobs = {}  # job id -> {status, done, total, info, error, csv}


@app.get("/")
def index():
    return send_from_directory(HERE, "sprint_frame_analyzer.html")


def work(job, video, side, sigma):
    try:
        def progress(done, total):
            jobs[job].update(done=done, total=total)
        info = track_video(video, os.path.join(WORK, job + ".csv"), os.path.join(WORK, job + "_fps.txt"),
                           side, sigma, progress)
        jobs[job].update(status="done", info=info, csv=info["csv"])
    except Exception as e:
        jobs[job].update(status="error", error=str(e))


@app.post("/track")
def track():
    f = request.files.get("video")
    if not f:
        return jsonify(error="No video received"), 400
    job = uuid.uuid4().hex[:8]
    path = os.path.join(WORK, job + os.path.splitext(f.filename)[1])
    f.save(path)
    side = request.form.get("near_side") or None
    sigma = float(request.form.get("sigma") or 1.5)
    jobs[job] = dict(status="running", done=0, total=0)
    threading.Thread(target=work, args=(job, path, side, sigma), daemon=True).start()
    return jsonify(job=job)


@app.get("/status/<job>")
def status(job):
    if job not in jobs:
        return jsonify(status="error", error="Unknown job"), 404
    return jsonify({k: v for k, v in jobs[job].items() if k != "csv"})


@app.get("/csv/<job>")
def csv_file(job):
    return send_file(jobs[job]["csv"], mimetype="text/csv")


if __name__ == "__main__":
    app.run(port=5000)
