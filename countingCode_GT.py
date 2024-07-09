import time
import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict
from utils import point_position
from pynput import keyboard
import threading

path_to_model = "D:\\omer\\tracking\\yolov8x_headCounting.pt"
# csv to write results
csv_filepath = "GT_omer_gate1_human_count_timestamps.csv"
vid_path = 'D:\\omer\\Human_counting_project\\gateVideos\\gate1.mp4'

# Define line coordinates
x1, y1, x2, y2 = 1252, 599, 2404, 612

class KeyCounter:
    def __init__(self):
        self.key1_count = 0  # Counter for key 'a'
        self.key2_count = 0  # Counter for key 'b'

    def on_press(self, key):
        try:
            # Record the key press
            # if key.char == 'z':
            #     self.key1_count += 1
            #     print(f"Key 'a' pressed {self.key1_count} times.")
            # elif key.char == 'x':
            #     self.key2_count += 1
            #     print(f"Key 'b' pressed {self.key2_count} times.")
            vk = key.vk  # Get the virtual key code
            # You will have to replace these with the VK codes for your specific numpad keys.
            if vk == 100:          #vk code for numpad4
                self.key2_count += 1
                print(f"Human_out {self.key2_count}")
            elif vk == 102:        #vk code for numpad6
                self.key1_count += 1
                print(f"Human_in {self.key1_count}")
        except AttributeError:
            pass  # Handle special keys

    def run(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()


key_counter = KeyCounter()
    
# Create a new thread for key listening
thread = threading.Thread(target=key_counter.run)
thread.daemon = True  # Daemon threads exit when the program does
thread.start()


import csv
with open(csv_filepath, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'count_in', 'count_out', 'count_in_GT', 'count_out_GT'])  # Write headers

def write_to_csv(timestamp, count_in, count_out, count_in_GT, count_out_GT):
    with open(csv_filepath, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, count_in, count_out, count_in_GT, count_out_GT])

# Initialize the model
model = YOLO(path_to_model)


# x1, y1, x2, y2 = 703, 72, 2491, 941   #gate 2
line_point1 = (x1, y1)
line_point2 = (x2, y2)

# Frame skip rate
frame_skip = 2

# Tracking history and count dictionary
track_history = defaultdict(lambda: [])
count_dict = {'in': 0, 'out': 0}

# Threshold for determining direction confidence
threshold = 0.6
cooldown_dict = {}  # Dictionary to store cooldown timers for each object
cooldown_time = 1200  # Cooldown time in frames


cv2.namedWindow('track', cv2.WINDOW_NORMAL)
cv2.resizeWindow("track", (1920, 1080))
num_frames=0

time_str="11:15:00"
hours, minutes, seconds = map(int, time_str.split(':'))
starting_time = hours * 3600 + minutes * 60 + seconds
current_time = starting_time  # Initial timestamp
# Main loop
i=1
while i<=1:
    try:
        i+=1
        start_time = time.time()
        cap = cv2.VideoCapture(vid_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('length: ', length )
        print('cap: ', cap)
        _fps= int(cap.get(cv2.CAP_PROP_FPS))
        print("_fps: ", _fps)
        frames_per_15_seconds = (15 * _fps) / 2
        frames_since_last_write = 0

        while True:
            ret, frame = cap.read()
            
            num_frames += 1
            if num_frames % frame_skip != 0:
                continue
            frames_since_last_write+=1

            results = model.track(frame, verbose=False, persist=True, conf=0.3, iou=0.8, show_conf=False, show_labels=False)
            try:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
            except Exception as e:
                continue
            
            # ano_frame = results[0].plot()
            ano_frame = frame
            cv2.line(ano_frame, line_point1, line_point2, (0, 0, 255), 3)  # draw a line
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                if cooldown_dict.get(track_id, 0) > 0:
                    cooldown_dict[track_id] -= 1
                    continue

                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) >= 2:  # Need at least two points to determine direction
                    pos_first = point_position(x1, y1, x2, y2, track[0][0], track[0][1])
                    pos_last = point_position(x1, y1, x2, y2, track[-1][0], track[-1][1])

                    # Check the direction of the object
                    if pos_first == "below" and pos_last == "above":
                        count_dict['out'] += 1
                        cooldown_dict[track_id] = cooldown_time
                    elif pos_first == "above" and pos_last == "below":
                        count_dict['in'] += 1
                        cooldown_dict[track_id] = cooldown_time

                    if len(track) > 30:  # Limit track history
                        track.pop(0)
                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(ano_frame, [points], isClosed=False, color=(0, 255,0), thickness=3)
            
            display_count_text = ", ".join([f"{k}: {v}" for k, v in count_dict.items()])
            cv2.putText(ano_frame, f"Counts: {display_count_text}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 214, 10), 3)
            if frames_since_last_write >= frames_per_15_seconds:
                current_time += 15
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(current_time))
                with open(csv_filepath, "a", newline='') as file:
                    write_to_csv(formatted_time, count_dict['in'], count_dict['out'], key_counter.key2_count, key_counter.key1_count)
                frames_since_last_write = 0
            cv2.imshow('track', ano_frame)
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        cap.release()
        track_history = defaultdict(lambda: [])
        cooldown_dict = {}
        num_frames = 0
        time.sleep(1)  # Rest for 2.5 minutes
        continue

cv2.destroyAllWindows()
