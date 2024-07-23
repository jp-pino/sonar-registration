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
    WIDTH = None
    HEIGHT = None
    RATIO = None
    frames = sorted(glob.glob(path), key=get_order)
    print(f"Found {len(frames)} frames")


    for frame in frames:
        try:
            frame = cv2.imread(frame)

            if WIDTH is None:
                HEIGHT, WIDTH, _ = frame.shape
                RATIO = HEIGHT / WIDTH

            height, width, layers = frame.shape

            if width != WIDTH or height != HEIGHT:
                # print(f"Resizing frame {frame.shape} to {WIDTH}x{HEIGHT}")
                padding = height - width * RATIO
                if padding >= 0:
                    frame = cv2.copyMakeBorder(frame, 0, 0, int(padding // 2), int(padding // 2), cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])
                else:
                    padding = width - height / RATIO
                    frame = cv2.copyMakeBorder(frame, int(padding // 2), int(padding // 2), 0, 0, cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])



                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            if video is None:
                print(f"Creating video with shape {width}x{height} at {fps} fps")
                print(f"\033[s")
                video = cv2.VideoWriter(name, 0x7634706d, fps, (WIDTH, HEIGHT))
            video.write(frame)

            # print(f"\033[uProcessed frame {frame.shape} {frame.shape[1]}x{frame.shape[0]}")
        except Exception as e:
            print(f"Error processing frame {id}: {e}")
            break

    cv2.destroyAllWindows()
    video.release()
