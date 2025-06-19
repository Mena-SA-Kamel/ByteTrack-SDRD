# Building on notebook used in to use FasterRCNN detections:
# https://www.datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box
import os
import ast
import cv2
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import torch
from yolox.tracker.byte_tracker import BYTETracker

MIN_THRESHOLD = 0.001
text_scale = 1
text_thickness = 2
line_thickness = 2
TEXT_COLOR = (0, 0, 255)
ID2CLASSES = {0: "hole"}
COLORS = [(0, 0, 255)]
CLASSES = ID2CLASSES.values()


class ByteTrackArgument:
    track_thresh = 0.5  # High_threshold
    track_buffer = 500  # Number of frame lost tracklets are kept
    match_thresh = 0.9  # Matching threshold for first stage linear assignment
    mot20 = False  # If used, bounding boxes are not clipped.


def calculate_centroid(tl_x, tl_y, w, h):
    mid_x = int(tl_x + w / 2)
    mid_y = int(tl_y + h / 2)
    return mid_x, mid_y


def convert_history_to_dict(track_history):
    history_dict = {}
    for frame_content in track_history:
        obj_ids, tlwhs, _ = frame_content
        for obj_id, tlwh in zip(obj_ids, tlwhs):
            tl_x, tl_y, w, h = tlwh
            mid_x, mid_y = calculate_centroid(tl_x, tl_y, w, h)

            if obj_id not in history_dict.keys():
                history_dict[obj_id] = [[mid_x, mid_y]]
            else:
                history_dict[obj_id].append([mid_x, mid_y])

    return history_dict


def get_color_by_id(track_id):
    # Generate a unique color for each ID
    np.random.seed(track_id)  # Ensure consistency per ID
    return tuple(int(x) for x in np.random.randint(0, 255, 3))


def plot_tracking(image, track_history):
    obj_ids, tlwhs, class_ids = track_history[-1]
    history_dict = convert_history_to_dict(track_history)

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    num_detections = len(tlwhs)
    label_count = {class_name: 0 for class_name in CLASSES}
    for label_idx in class_ids:
        label_count[ID2CLASSES[label_idx]] += 1

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        class_id = class_ids[i]
        id_text = "{}".format(obj_id)
        color = get_color_by_id(obj_id)  # Use unique color per ID

        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
        )
        cv2.putText(
            im,
            id_text,
            (intbox[0], intbox[1]),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            color,
            thickness=text_thickness,
        )
        cv2.putText(
            im,
            ID2CLASSES[class_id],
            (intbox[0], intbox[3] + 20),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            color,
            thickness=text_thickness,
        )

        for idx in range(len(history_dict[obj_id]) - 1):
            prev_point, next_point = (
                history_dict[obj_id][idx],
                history_dict[obj_id][idx + 1],
            )
            cv2.line(im, prev_point, next_point, color, 2)

    return im


def df_to_detections_tensor(df):
    """
    Convert a detection dataframe to a torch.Tensor for BYTETrack.

    Parameters:
        df (pd.DataFrame): Must have 'boxes' (xyxy) and 'scores' columns.

    Returns:
        torch.Tensor: (N, 5) detections [x, y, w, h, score]
    """

    detections = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = ast.literal_eval(row["boxes"])
        score = row["scores"]
        detections.append([x1, y1, x2, y2, score])

    if len(detections) == 0:
        return torch.empty((0, 5), dtype=torch.float64)

    return torch.tensor(detections, dtype=torch.float64)


def run_bytetrack(video_path, detections_dir, save_path, classes_to_track=[]):
    # Loading the video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (int(width), int(height))
    )

    tracker = BYTETracker(ByteTrackArgument)

    info_imgs = [height, width]
    img_size = [height, width]
    history = deque()

    # Use tqdm to create a progress bar for the frame loop
    with tqdm(
        total=total_frames,
        desc="Bytetrack on Frames",
        unit="frame",
    ) as pbar:
        # Loop through each frame
        frame_ix = 0
        while True:
            ret, frame = cap.read()  # Read a frame

            if not ret:
                break  # Break if the video ends

            det_df = pd.read_csv(os.path.join(detections_dir, f"frame_{frame_ix}.csv"))
            holes_df = det_df[det_df.labels.isin(classes_to_track)]
            holes_tensor = df_to_detections_tensor(holes_df)
            if len(holes_df) < 9:
                frame_ix += 1
                pbar.update(1)
                frame = np.ascontiguousarray(np.copy(frame))
                vid_writer.write(frame)
                continue
            online_targets = tracker.update(holes_tensor, info_imgs, img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_classes = []

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                online_ids.append(tid)
                online_tlwhs.append(tlwh)
                online_classes.append(0)
            # print(sorted(online_ids))
            if len(history) < 200:
                history.append((online_ids, online_tlwhs, online_classes))
            else:
                history.popleft()
                history.append((online_ids, online_tlwhs, online_classes))

            frame = plot_tracking(frame, history)
            vid_writer.write(frame)
            frame_ix += 1

            pbar.update(1)
    cap.release()


if __name__ == "__main__":
    detections_dir = "path/to/fasterrcnn/detections"
    video_path = "path/to/video"
    save_path = "path/to/save/video/w/tracks"
    classes_to_track = ["class_A", "class_B"]
    run_bytetrack(video_path, detections_dir, save_path, classes_to_track)
