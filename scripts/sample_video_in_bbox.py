import cv2
import os
import random
import argparse

# --- Mouse callback for selecting bounding box ---
bbox_points = []
cursor_pos = None  # track current cursor position


def select_bbox(event, x, y, flags, param):
    global bbox_points, cursor_pos
    cursor_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_points.append((x, y))
        print(f"Point selected: {x}, {y}")


def main(video_path, out_dir, n_samples=10):
    global bbox_points, cursor_pos

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames.")

    # Grab a frame near the end for bounding box selection
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total_frames - 50, 0))
    ret, frame = cap.read()
    if not ret:
        raise IOError("Could not read frame for bounding box selection.")

    # Show frame for bbox selection
    cv2.namedWindow("Select Bounding Box (UL then LR)")
    cv2.setMouseCallback("Select Bounding Box (UL then LR)", select_bbox)

    while True:
        disp = frame.copy()

        # Draw crosshair guide lines if cursor is inside the window
        if cursor_pos is not None:
            x, y = cursor_pos
            color = (255, 255, 255)  # white
            thickness = 1
            # vertical line
            cv2.line(disp, (x, 0), (x, disp.shape[0]), color, thickness)
            # horizontal line
            cv2.line(disp, (0, y), (disp.shape[1], y), color, thickness)

        cv2.imshow("Select Bounding Box (UL then LR)", disp)
        if cv2.waitKey(20) & 0xFF == 27:  # ESC to cancel
            cv2.destroyAllWindows()
            return
        if len(bbox_points) == 2:
            print("got bbox points:", bbox_points)
            cv2.destroyAllWindows()
            break

    # Extract bounding box
    (x1, y1), (x2, y2) = bbox_points
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")

    # Pick random frame indices from the second half of the video
    frame_indices = random.sample(range(total_frames // 2, total_frames), n_samples)
    print(f"Sampling frames: {frame_indices}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {idx}")
            continue

        crop = frame[y1:y2, x1:x2]
        out_name = f"{base_name}_f{idx}.png"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, crop)
        print(f"Saved {out_path}")

    cap.release()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample cropped frames from a video.")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("out_dir", help="Directory to save cropped frames")
    parser.add_argument("--n", type=int, default=10, help="Number of frames to sample")
    args = parser.parse_args()

    main(args.video_path, args.out_dir, args.n)
