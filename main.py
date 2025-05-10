import numpy as np
import torch
import os
from ultralytics import YOLO
from tracker import Tracker
from collections import defaultdict
from deep_sort.deep_sort.detection import Detection
import cv2
import csv
import random
from reid_utils import FaceEmbedder
from cross_video_matcher import CrossVideoMatcher

OUTPUT_CSV = 'final_results.csv'
DEBUG_FOLDER = './debug_faces'

coco_model = YOLO('yolov8n.pt')
face_embedder = FaceEmbedder()

input_folder = './input_videos'
output_folder = './output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

all_tracks = []
video_specific_tracks = {}
colors = [(random.randint(0, 255), random.randint(0, 255),
           random.randint(0, 255)) for _ in range(100)]


matcher = CrossVideoMatcher(similarity_threshold=0.6)


for video_file in os.listdir(input_folder):
    if not video_file.endswith(('.mp4', '.avi', '.mov')):
        continue

    print(f"Processing video: {video_file}")
    input_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, f'out_{video_file}')
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *'MP4V'), fps, (width, height))

    frame_idx = 0
    tracker = Tracker()
    video_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = coco_model(frame)[0]
        detections = []

        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = det
            if int(cls) == 0 and score > 0.5:
                bbox = np.array([x1, y1, x2 - x1, y2 - y1])
                detections.append(Detection(bbox, score, np.zeros(128)))

        tracks = tracker.update(frame, detections)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            l, t = max(0, int(l)), max(0, int(t))
            r, b = min(width, int(r)), min(height, int(b))

            if r - l < 20 or b - t < 20:
                continue

            bbox = [l, t, r - l, b - t]
            person_img = frame[t:b, l:r]

            if person_img.size == 0:
                continue

            # Save cropped face for debugging
            debug_path = os.path.join(
                DEBUG_FOLDER, f'{video_file}_frame{frame_idx}_track{track_id}.jpg')
            cv2.imwrite(debug_path, person_img)

            embedding = face_embedder.get_embedding(person_img)
            if embedding is None or np.linalg.norm(embedding) < 0.1:
                print(
                    f"[Warning] Skipping invalid embedding at frame {frame_idx} in {video_file}")
                continue

            video_tracks.append({
                'track_id': track_id,
                'video': video_file,
                'frame': frame_idx,
                'bbox': bbox,
                'embedding': embedding
            })

            # Show temporary local ID
            color_idx = track_id % len(colors)
            cv2.rectangle(
                frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), colors[color_idx], 2)
            cv2.putText(frame, f'Track: {track_id}', (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[color_idx], 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")

    cap.release()
    out.release()

    video_specific_tracks[video_file] = video_tracks
    all_tracks.extend(video_tracks)
    print(f"Finished {video_file} with {len(video_tracks)} valid tracks")

print(f"Total tracks collected: {len(all_tracks)}")


print("Performing cross-video matching...")
final_results = matcher.match_tracks(all_tracks)


for video_file in video_specific_tracks:
    input_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, f'final_{video_file}')
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *'MP4V'), fps, (width, height))

    frame_results = defaultdict(list)
    for r in final_results:
        if r['video'] == video_file:
            frame_results[r['frame']].append(r)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for r in frame_results[frame_idx]:
            bbox = r['bbox']
            pid = r['final_id']
            color_idx = pid % len(colors)
            cv2.rectangle(
                frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), colors[color_idx], 2)
            cv2.putText(frame, f'ID: {pid}', (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[color_idx], 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

# === Save CSV ===
print(f"Saving results to {OUTPUT_CSV}")
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'video', 'frame', 'bbox_x',
                    'bbox_y', 'bbox_w', 'bbox_h'])
    for r in final_results:
        writer.writerow([r['final_id'], r['video'], r['frame'], *r['bbox']])

print("All processing complete.")
