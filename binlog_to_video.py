import os.path
import sys
import cv2
import glob
import numpy as np
from ray.util.client import ray

from utils import binlog
from utils.image import to_fan, extract_data_and_mask



class FanModule():
    def __init__(self, bearings, cache="cache"):
        super().__init__()
        self.bearings = bearings
        self.mapping = None
        self.FOV = None
        self.HEIGHT = None
        self.WIDTH = None
        self.mask = None
        self.cache = cache
        self.init_origin = False

        # Make sure cache directory exists
        os.makedirs(self.cache, exist_ok=True)

    @staticmethod
    def get_fan(raw, height, width, mapping):
        if raw is None:
            return None, None

        fan = np.zeros((height, width))
        for x in range(width):
            for y in range(height):
                r, beam_id = mapping[y, x]
                try:
                    if 0 < r < raw.shape[0] and 0 < beam_id < raw.shape[1]:
                        fan[y, x] = raw[r, beam_id]
                except Exception as e:
                    print(f"beam: {beam_id}, r: {r} -> x: {x}, y: {y}")
                    print(f"raw shape: {raw.shape}, fan shape: {fan.shape}")
                    raise e

        return fan


    def get_bearing(self, theta, width):
        index = np.argmin(np.abs(self.bearings - theta))
        return int(index * width / len(self.bearings))

    def run(self, a):
        FOV = np.max(self.bearings) - np.min(self.bearings)
        HEIGHT = a.shape[0]
        WIDTH = int(2 * HEIGHT * np.sin(np.deg2rad(FOV / 2)))
        mask = None

        print(f"    > FOV: {FOV}, HEIGHT: {HEIGHT}, WIDTH: {WIDTH}, Number of bearings: {len(self.bearings)}, RAW Width: {a.shape[1]}, RAW Height: {a.shape[0]}")
        if (self.mapping is None or FOV != self.FOV
                or HEIGHT != self.HEIGHT
                or WIDTH != self.WIDTH):
            self.FOV = FOV
            self.HEIGHT = HEIGHT
            self.WIDTH = WIDTH
            if os.path.exists(os.path.join(self.cache, f"{FOV}_{WIDTH}_{HEIGHT}_mapping.npy")):
                print(f"    > Loading mapping from cache")
                self.mapping = np.load(os.path.join(self.cache, f"{FOV}_{WIDTH}_{HEIGHT}_mapping.npy"))
                self.mask = np.load(os.path.join(self.cache, f"{FOV}_{WIDTH}_{HEIGHT}_mask.npy"))
            else:
                print(f"    > Recalculating self.mapping")
                print(f"    > self.bearings: {self.bearings}")
                print(f"    > FOV: {FOV}, HEIGHT: {HEIGHT}, WIDTH: {WIDTH}")
                self.mapping = np.zeros((HEIGHT, WIDTH, 2), dtype=int)
                for x in range(WIDTH):
                    for y in range(HEIGHT):
                        theta = np.rad2deg(np.arctan2(x - WIDTH / 2, HEIGHT - y))
                        r = np.sqrt((x - WIDTH / 2) ** 2 + (HEIGHT - y) ** 2)
                        beam_id = -1
                        if np.min(self.bearings) < theta < np.max(self.bearings):
                            beam_id = self.get_bearing(theta, a.shape[1])
                        self.mapping[y, x] = [r, beam_id]
                print(f"    > Mapping shape: {self.mapping.shape}")

                np.save(os.path.join(self.cache, f"{FOV}_{WIDTH}_{HEIGHT}_mapping.npy"), self.mapping)

                self.mask = None
                mask = self.get_fan(np.ones_like(a), self.HEIGHT, self.WIDTH, self.mapping)

                np.save(os.path.join(self.cache, f"{FOV}_{WIDTH}_{HEIGHT}_bearings.npy"), self.bearings)

        print(f"    > Running fan module")
        a = self.get_fan(a, self.HEIGHT, self.WIDTH, self.mapping)

        if mask is not None:
            print(f"    > Mask shape: {mask.shape}")
            self.mask = mask
            np.save(os.path.join(self.cache, f"{FOV}_{WIDTH}_{HEIGHT}_mask.npy"), self.mask)

        print(f"    > Fan shape: {a.shape}")
        return a

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

    fan_module = None

    video = None
    count = 0
    for ping, raw, ts in binlog.read_ping_2(path):
        try:
            if fan_module is None:
                fan_module = FanModule(bearings=ping.bearings)
                print(f"Fan module initialized")

            frame = fan_module.run(raw)
            print(f"Frame shape: {frame.shape}")
            # frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
            # from 0 - 1 to 0 - 255
            # frame = frame * 255
            # frame = frame.astype(np.uint8)

            height, width = frame.shape
            print(f"Frame shape: {frame.shape}")
            print(f"Widht: {width}, Height: {height}")
            print(f"min: {np.min(frame)}, max: {np.max(frame)}")

            if video is None:
                print(f"Creating video with shape {width}x{height} at {fps} fps")
                video = cv2.VideoWriter(name, 0X00000021, fps, (width, height), False)

            print(f"Processing ping")
            frame = cv2.putText(frame.copy(), f"Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video.write(frame)

            if max_frames != None and count >= max_frames:
                break
            count += 1
        except Exception as e:
            print(f"Error processing frame: {e}")
            break

    cv2.destroyAllWindows()
    video.release()
