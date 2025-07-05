import select

# Patch select to avoid WinError 10038
_ORIGINAL_SELECT = select.select

def _safe_select(*args, **kwargs):
    try:
        return _ORIGINAL_SELECT(*args, **kwargs)
    except OSError as e:
        if e.winerror == 10038:
            return [], [], []
        raise

select.select = _safe_select

import cv2
import math
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

POINTS = {
    "shoulder": 12,
    "hip": 24,
    "knee": 26,
    "ankle": 28,
    "toe": 32,
    "left_ankle": 27,
    "left_toe": 31
}

UNIFORM_COLOR = (0, 255, 255)

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return np.array([]), [], {}, None, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    keypoints, frames = [], []
    trajectories = {name: [] for name in POINTS}
    pixel_to_meter = None
    frame_width, frame_height = 0, 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_width == 0:
                frame_height, frame_width = frame.shape[:2]

            frames.append(frame.copy())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                kp = np.array([[p.x, p.y] for p in lm])
                keypoints.append(kp)

                if pixel_to_meter is None:
                    hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    leg_length_px = abs(hip.y - ankle.y) * frame_height

                    if leg_length_px > 10:
                        pixel_to_meter = (0.45 * 1.7) / leg_length_px
                        print(f"Calibrated: 1px = {pixel_to_meter:.6f}m")
                    else:
                        pixel_to_meter = 0.001
                        print("Using default calibration")

                for name, idx in POINTS.items():
                    landmark = lm[idx]
                    if landmark.visibility > 0.5:
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        trajectories[name].append((x, y))
                    else:
                        if trajectories[name]:
                            trajectories[name].append(trajectories[name][-1])
                        else:
                            trajectories[name].append((0, 0))
            else:
                for name in POINTS:
                    if trajectories[name]:
                        trajectories[name].append(trajectories[name][-1])
                    else:
                        trajectories[name].append((0, 0))
    finally:
        cap.release()

    if pixel_to_meter is None:
        print("Warning: Using default calibration")
        pixel_to_meter = 0.001

    return np.array(keypoints), frames, trajectories, pixel_to_meter, fps

def angle(a, b, c):
    ab, bc = b - a, c - b
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def count_colliding_steps(right_ankle_traj, left_ankle_traj, threshold=20, cooldown_seconds=0.6, fps=30):
    """
    Improved: Count steps using ankle collisions with a longer cooldown to avoid double-counting.
    Each collision is a step (one footfall).
    """
    step_frames = []
    cooldown_frames = int(cooldown_seconds * fps)
    last_step_frame = -cooldown_frames

    for i in range(min(len(right_ankle_traj), len(left_ankle_traj))):
        if right_ankle_traj[i] == (0, 0) or left_ankle_traj[i] == (0, 0):
            continue
        dist = math.dist(right_ankle_traj[i], left_ankle_traj[i])
        if dist <= threshold and (i - last_step_frame) >= cooldown_frames:
            step_frames.append(i)
            last_step_frame = i
    return step_frames

def calculate_gait_parameters(keypoints, trajectories, pixel_to_meter, fps):
    if keypoints.size == 0 or len(trajectories["ankle"]) < 2:
        return np.zeros(6), [], [], []

    right_ankle_traj = trajectories["ankle"]
    left_ankle_traj = trajectories["left_ankle"]

    right_arr = np.array(right_ankle_traj)
    left_arr = np.array(left_ankle_traj)
    valid = (right_arr != (0, 0)).all(axis=1) & (left_arr != (0, 0)).all(axis=1)
    right_arr = right_arr[valid]
    left_arr = left_arr[valid]

    if len(right_arr) == 0 or len(left_arr) == 0:
        return np.zeros(6), [], [], []

    pixel_dists = np.linalg.norm(right_arr - left_arr, axis=1)
    stride_length = np.max(pixel_dists) * pixel_to_meter

    # Improved step detection (avoid double-counting)
    step_events = count_colliding_steps(trajectories["ankle"], trajectories["left_ankle"], fps=fps)
    total_steps = len(step_events)

    # Cadence: steps per minute
    valid_pose_frames = np.count_nonzero([pt != (0, 0) for pt in trajectories["ankle"]])
    total_time_sec = valid_pose_frames / fps if fps > 0 else 0

    cadence = ((total_steps +1) / total_time_sec) * 60 if total_time_sec > 0 else 0

    speed_kmph = stride_length * cadence / 120 * 3.6 if cadence > 0 else 0

    knee_angles, hip_angles, ankle_angles = [], [], []
    for i in range(len(keypoints)):
        hip = keypoints[i, POINTS["hip"]]
        knee = keypoints[i, POINTS["knee"]]
        ankle = keypoints[i, POINTS["ankle"]]
        toe = keypoints[i, POINTS["toe"]]
        shoulder = keypoints[i, POINTS["shoulder"]]
        knee_angles.append(angle(hip, knee, ankle))
        hip_angles.append(angle(shoulder, hip, knee))
        ankle_angles.append(angle(knee, ankle, toe))

    return np.array([
        stride_length,
        cadence,
        np.mean(knee_angles),
        np.mean(hip_angles),
        np.mean(ankle_angles),
        speed_kmph
    ]), [knee_angles, hip_angles, ankle_angles], step_events

def overlay_text_on_video(frames, gait_params, suggestions, joint_angles,
                          output_path, trajectories, step_events):
    if not frames:
        print("Error: No frames to process")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    step_events_list = step_events

    for i, frame in enumerate(frames):
        disp = frame.copy()

        for name, points in trajectories.items():
            if name == "left_ankle":
                continue
            for j in range(1, min(i+1, len(points))):
                if points[j-1] != (0,0) and points[j] != (0,0):
                    alpha = max(0.3, j/len(points))
                    col = tuple(int(c*alpha) for c in UNIFORM_COLOR)
                    cv2.line(disp, points[j-1], points[j], col, 2)

        for name, points in trajectories.items():
            if name == "left_ankle":
                continue
            if i < len(points) and points[i] != (0,0):
                x, y = points[i]
                cv2.circle(disp, (x, y), 6, UNIFORM_COLOR, -1)
                cv2.circle(disp, (x, y), 8, (255, 255, 255), 2)

        if i in step_events_list:
            right_ankle_pos = trajectories["ankle"][i]
            left_ankle_pos = trajectories["left_ankle"][i]
            if right_ankle_pos != (0,0):
                cv2.circle(disp, right_ankle_pos, 12, (0, 0, 255), 3)
            if left_ankle_pos != (0,0):
                cv2.circle(disp, left_ankle_pos, 12, (255, 0, 0), 3)

        y_offset = 30
        if i < len(joint_angles[0]):
            knee_angle = joint_angles[0][i]
            hip_angle = joint_angles[1][i]
            ankle_angle = joint_angles[2][i]
        else:
            knee_angle = joint_angles[0][-1] if joint_angles[0] else 0
            hip_angle = joint_angles[1][-1] if joint_angles[1] else 0
            ankle_angle = joint_angles[2][-1] if joint_angles[2] else 0

        params_text = [
            f"Stride Length: {gait_params[0]:.4f} m",
            f"Cadence: {gait_params[1]:.2f} steps/min",
            f"Hip Angle: {hip_angle:.1f} deg",
            f"Knee Angle: {knee_angle:.1f} deg",
            f"Ankle Angle: {ankle_angle:.1f} deg",
            f"Speed: {gait_params[5]:.2f} km/h",
            f"Steps: {len(step_events)+1}"
        ]

        for text in params_text:
            cv2.putText(disp, text, (20, y_offset), font, 0.6, (0, 0, 0), 3)
            cv2.putText(disp, text, (20, y_offset), font, 0.6, (0, 255, 0), 2)
            y_offset += 25

        for idx, suggestion in enumerate(suggestions):
            cv2.putText(disp, suggestion, (20, height - 40 - idx * 25),
                        font, 0.6, (0, 0, 0), 3)
            cv2.putText(disp, suggestion, (20, height - 40 - idx * 25),
                        font, 0.6, (0, 255, 255), 2)

        out.write(disp)

    out.release()

def generate_suggestions(gait_params):
    suggestions = []
    OPTIMAL_VALUES = {
        "stride_length": (0.6, 1.0),
        "cadence": (70, 120),
        "knee_angle": (10, 70),
        "hip_angle": (10, 40),
        "ankle_angle": (10, 30)
    }
    param_names = ["Stride Length", "Cadence", "Knee Angle", "Hip Angle", "Ankle Angle"]

    for i, (param, (low, high)) in enumerate(OPTIMAL_VALUES.items()):
        value = gait_params[i]
        if value < low:
            suggestions.append(f"Increase {param_names[i]} to improve gait.")
        elif value > high:
            suggestions.append(f"Decrease {param_names[i]} to improve posture.")

    if not suggestions:
        suggestions.append("Gait parameters are within optimal range!")

    return suggestions

def process_video(video_path, output_path="output_video.mp4"):
    try:
        MODEL_PATH = "gait_model.pkl"
        DATA_PATH = "training_data.pkl"

        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            model = RandomForestRegressor(warm_start=True)

        if os.path.exists(DATA_PATH):
            X_train, y_train = joblib.load(DATA_PATH)
        else:
            X_train, y_train = [], []

        keypoints, frames, trajectories, pixel_to_meter, fps = extract_keypoints(video_path)

        if keypoints.size == 0 or not frames:
            raise Exception("No keypoints or frames extracted from video")

        gait_params, joint_angles, step_events = calculate_gait_parameters(
            keypoints, trajectories, pixel_to_meter, fps
        )

        suggestions = generate_suggestions(gait_params)

        print(f"Detected {len(step_events)} steps")
        print(f"Stride Length: {gait_params[0]:.4f}m")
        print(f"Cadence: {gait_params[1]:.2f} steps/min")
        print(f"Speed: {gait_params[5]:.2f} km/h")

        overlay_text_on_video(
            frames,
            gait_params,
            suggestions,
            joint_angles,
            output_path,
            trajectories,
            step_events
        )

        if joint_angles[0]:
            knee_mean = np.mean(joint_angles[0])
            if not np.isnan(knee_mean):
                X_train.append(gait_params.tolist())
                y_train.append(knee_mean)
                model.fit(np.array(X_train), np.array(y_train))
                joblib.dump(model, MODEL_PATH)
                joblib.dump((X_train, y_train), DATA_PATH)
                print("Model updated and saved")

        return gait_params
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

#Example use
video_path = "sample.mp4"
process_video(video_path)

try:
    from google.colab import files
    files.download("output_video.mp4")
    print("Video download initiated.")
except ImportError:
    print("Not running in Colab. Output video saved locally.")
