# Sprint-Pose-Tracker

# Sprint Pose Tracker and Frame Analyzer

Tools for measuring sprint mechanics from video. A Python tracker finds 13 body joints in every frame and writes them to CSV. A browser-based analyzer lets you step through the footage, fix any mis-tracked points, mark events like touchdown and toe-off, and measure joint angles. A Flask server connects the pieces.

The goal is to turn ordinary sprint video into clean, per-frame biomechanics data, starting with top-speed running and extending to acceleration.



## What it does

- **Tracks 13 joints per frame** with MediaPipe Pose and OpenCV: head, neck, elbow and wrist (front and rear), pelvis, and knee, heel, and toe (front and rear). "Front" means the near-side limbs, assigned using hip depth in the first frame.
- **Smooths lightly** before saving.
- **Exports CSV** to `sprint_coordinates.csv` with the columns `frame, joint_name, joint_number, x, y`, plus a text file with the video's fps.
- **Frame analyzer (HTML):** loads the video, CSV, and fps; steps frame by frame; lets you drag points to correct them; marks events; includes a three-point angle calculator; and exports a corrected CSV (with a flag marking manually edited points) plus a separate events CSV.

## Project files

| File | Role |
| --- | --- |
| `sprint_pose_tracker_csv.py` | Runs MediaPipe Pose on a video and writes the joint CSV and fps file |
| `sprint_frame_analyzer.html` | Single-file viewer for reviewing and correcting the tracked points |
| `[server filename].py` | Flask server: [one sentence on what the server does] |

## Requirements

- Python 3
- A modern web browser
- Libraries:

```
pip install mediapipe opencv-python flask
```

## Usage

1. **Track a video.** Run the tracker on a sprint clip:
   ```
   python sprint_pose_tracker_csv.py [add the arguments you use, e.g. the video path]
   ```
   This writes `sprint_coordinates.csv` and an fps text file.
2. **Start the server:**
   ```
   python [server filename].py
   ```
   Then open the address Flask prints in the terminal (usually `http://127.0.0.1:5000`).
3. **Review in the analyzer.** Load the video, the CSV, and the fps value. Step through frames, drag any mis-tracked points into place, and mark events such as touchdown and toe-off.
4. **Export.** Save the corrected CSV and the events CSV for analysis.

## Known limitations

Tracking errors seen so far, all of which the analyzer's drag-to-correct tool is meant to handle:

- The rear leg crossing into the front leg
- Unstable output in the first seconds, when no athlete is in frame yet
- Lower-leg points fluttering around toe-off

## Results

Test footage: a slow-motion sprint clip (about 96 fps).

- Frames tracked: [N]
- Frames needing manual correction: [X]% (count the flagged points in the corrected CSV)

## Roadmap

Planned or in progress:

- Automatic detection of touchdown and toe-off, with ground contact time
- Event data stored per event, reviewed and adjusted in the analyzer
- Upper-leg angular velocity, lower-leg flexion, hip height, and center of mass
- Acceleration-phase analysis (the tracker has mainly covered top speed so far)
- Optional athlete inputs, such as height, for scaling

## Notes

- Sample video is not included because of file size and privacy.

## Author

Zining Luo, Computer Science, UC Santa Barbara
