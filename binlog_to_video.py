import os.path
import sys
import cv2
import glob
import numpy as np

from utils import binlog
from utils.image import to_fan, extract_data_and_mask

if __name__ == "__main__":
    # Take path from command line
    path = sys.argv[1] if len(sys.argv) > 1 else "./logs/log-multibeam.bez"
    out = sys.argv[2] if len(sys.argv) > 2 else "./out/"
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 10
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None

    # extract the filename from the path
    name = os.path.join(out, f"{os.path.basename(path)}.mp4")

    # make sure the output directory exists
    os.makedirs(out, exist_ok=True)

    video = None
    count = 0
    for id, data, gain, aperture, ts in binlog.read_ping(path):
        try:
            frame, _ = extract_data_and_mask(to_fan(data, aperture))
            height, width = frame.shape

            if video is None:
                print(f"Creating video with shape {width}x{height} at {fps} fps")
                video = cv2.VideoWriter(name, 0x7634706d, fps, (width, height), False)

            print(f"Processing ping {id}")
            frame = cv2.putText(frame.copy(), f"Frame {id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video.write(frame)

            if max_frames != None and count >= max_frames:
                break
            count += 1
        except Exception as e:
            print(f"Error processing frame {id}: {e}")
            break

    cv2.destroyAllWindows()
    video.release()
