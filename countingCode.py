import csv

# File for maintaining the CSV
csv_file = "human_count_timestamps.csv"
path_to_videos = "D:\\omer\\tracking\\sample_videos\\**\\*"
path_to_human_model = "D:\\omer\\tracking\\yolov8x_headCounting.pt"
# Add a line for counting
count_line_position = 350  # Position (height) in the frame where the counting line is drawn

with open(csv_file, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'count_in', 'count_out'])  # Write headers

def write_to_csv(timestamp, count_in, count_out):
    with open(csv_file, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, count_in, count_out])

        # path_to_video = "D:\\omer\\tracking\\sample.mp4"
# path_to_human_model = "D:\\omer\\tracking\\yolov8n_headCounting.pt"
# path_to_vehicle_model = "D:\\omer\\tracking\\yolov5s_vehicle_detection.pt"

import glob
videos = glob.glob(path_to_videos, recursive=True)
videos = ["D:\\omer\\siddPhone.mp4"]
print(videos)
import time
from collections import defaultdict

import cv2
import numpy as np
import time
from ultralytics import YOLO
from supervision.tools.line_counter import LineCounter
import glob

def putTextMultiLine(img, text, org, fontFace, fontScale, color, thickness, lineType):
                line_spacing = 35  # Define the line spacing
                x, y = org
                for i, line in enumerate(text.split('\n')):
                    y_offset = i * line_spacing
                    cv2.putText(img, line, (x, y + y_offset), fontFace, fontScale, color, thickness, lineType)


# Add counters
counter_humans_in = 0
counter_humans_out = 0

# Add counters for vehicles
counter_vehicle_in = 0
counter_vehicle_out = 0

window = "YOLOv8 Tracking"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
# cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(window, (1080, 720))
# Load the YOLOv8 model
human_model = YOLO(path_to_human_model)
# vehicle_model = YOLO(path_to_vehicle_model)
count_ids = []

# Initialize variables
frame_count = 0

starting_time = 3 * 3600  # 3:00 PM in seconds from 12:00 AM
current_time = starting_time  # Initial timestamp

start_time=time.time()
for i, video in enumerate(videos):
    # Open the video file
    cap = cv2.VideoCapture(video)
    # Get the frames per second (fps) from the video capture object
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the number of frames that represent 15 seconds of video time
    frames_per_15_seconds = 15 * fps
    # Print the fps
    print("Frames per second (fps) (original):", fps)
    print('video no.', i, '/', len(videos))
    # Store the track history
    track_history_humans = defaultdict(lambda: [])
    track_history_vehicles = defaultdict(lambda: [])
    count_ids_humans = []
    count_ids_vehicles = []
    # Loop through the video frames
    frames_since_last_write = 0
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # Draw counting line

        if success:
            # Increment the frame count
            frame_count += 1
            frames_since_last_write += 1
            # if frame_count % 1 != 0:
            #     continue
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            # Run YOLOv8 tracking on the frame for humans
            human_results = human_model.track(frame, persist=True, verbose=False, device='cuda')
            # Run YOLOv8 tracking on the frame for vehicles
            # vehicle_results = vehicle_model.track(frame, persist=True, verbose=False, device='cpu')
            annotated_frame = frame
            cv2.line(annotated_frame, (0, count_line_position), (frame.shape[1], count_line_position), (0, 255, 0), 2)
            # Get the boxes and track IDs
            human_boxes = human_results[0].boxes.xywh.cpu()
            if human_results[0].boxes.id is not None:
                track_ids_humans = human_results[0].boxes.id.int().cpu().tolist()
                for track_id in track_ids_humans:
                    if track_id not in count_ids_humans:
                        count_ids_humans.append(track_id)
                # Visualize the results on the frame
                # annotated_frame = human_results[0].plot()

                # # Plot the tracks
                for box, track_id in zip(human_boxes, track_ids_humans):
                    x, y, w, h = box
                    track = track_history_humans[track_id]
                    
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=3)
                     # Count when the object crosses the line
                    if len(track) >= 2:
                        prev_point = track[-2]
                        curr_point = track[-1]
                        
                        # Check if the object has crossed the line
                        if prev_point[1] < count_line_position and curr_point[1] >= count_line_position:
                            counter_humans_in += 1
                            # print("Human_in:", counter_humans_in)
                            
                        elif prev_point[1] > count_line_position and curr_point[1] <= count_line_position:
                            counter_humans_out += 1
                            # print("Human_out:", counter_humans_out)

            # print(frame_count, fps)
            if frames_since_last_write >= frames_per_15_seconds:
                current_time += 15  # Increment by 15 seconds
                hours = int(current_time // 3600)
                minutes = int((current_time % 3600) // 60)
                seconds = int(current_time % 60)
                formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                write_to_csv(formatted_time, counter_humans_in, counter_humans_out)
                print('updating csv, ', counter_humans_in, counter_humans_out)
                frames_since_last_write = 0  # Reset the frame counter
            # This could be more sophisticated based on what you want to show
            annotated_frame = human_results[0].plot()
            # annotated_frame = vehicle_results[0].plot(img=annotated_frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Usage in your loop
            text_to_display = f'human_in: {counter_humans_in}\nhuman_out: {counter_humans_out}'
            putTextMultiLine(annotated_frame, text_to_display, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 6, cv2.LINE_4)
            # Display the annotated frame
            cv2.imshow(window, annotated_frame)

            # Calculate fps every 100 frames
            if frame_count % 100 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = frame_count / elapsed_time
                print("Frames per second (fps) after", frame_count, "frames:", fps)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    print(f"no. of people in video {video}\nin: ", counter_humans_in, 'out: ', counter_humans_out)
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    end_time = time.time()
print(start_time, end_time, end_time-start_time)

# print(len(count_ids))