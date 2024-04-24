import math
import os.path
import re
import sys
import cv2
import glob
from pathlib import Path

file_pattern = re.compile(r'.*?(\d+).*?')


def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


if __name__ == "__main__":
    # Take path from command line
    path = sys.argv[1] if len(sys.argv) > 1 else "./out/frames_*.png"
    name = sys.argv[2] if len(sys.argv) > 2 else "./out/video.mp4"
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 10

    # make sure the output directory exists
    os.makedirs(os.path.dirname(name), exist_ok=True)

    video = None
    frames = sorted(glob.glob(path), key=get_order)

    for frame in frames:
        try:
            frame = cv2.imread(frame)
            height, width, layers = frame.shape

            if video is None:
                print(f"Creating video with shape {width}x{height} at {fps} fps")
                video = cv2.VideoWriter(name, 0x7634706d, fps, (width, height))

            video.write(frame)
        except Exception as e:
            print(f"Error processing frame {id}: {e}")
            break

    cv2.destroyAllWindows()
    video.release()
