import argparse
import json
import cv2
import time
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import numpy as np

class EventLogger:
    def __init__(self):
        self.events = []

    def log(self, frame_num, track_id, event_type, bbox, conf):
        event = {
            "frame_number": int(frame_num),     
            "track_id": int(track_id),          
            "event_type": event_type,
            "bounding_box": bbox,
            "confidence": float(conf)
        }
        self.events.append(event)
        print(f"EVENT DETECTED: {event_type} by ID {track_id} at frame {frame_num}")

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.events, f, indent=4)

class ZoneManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.zones = []
        for z in config['zones']:
            self.zones.append({
                "name": z['name'],
                "polygon": Polygon(z['coordinates']),
                "pts": np.array(z['coordinates'], np.int32).reshape((-1, 1, 2)),
                "loiter_thresh": z['loitering_threshold_seconds']
            })
        
        self.fps = config.get('target_fps', 30)
        self.state = {} # track_id -> {zone_name: entry_frame}
        self.logged_loiter = set() # (track_id, zone_name)

    def process_point(self, frame_num, track_id, point, bbox, conf, logger):
        p = Point(point)
        
        if track_id not in self.state:
            self.state[track_id] = {}

        current_zones = []
        
        for zone in self.zones:
            zone_name = zone['name']
            
            # Check if point is in polygon
            if zone['polygon'].contains(p):
                current_zones.append(zone_name)
                
                # Zone Intrusion Logic
                if zone_name not in self.state[track_id]:
                    self.state[track_id][zone_name] = frame_num
                    logger.log(frame_num, track_id, f"Intrusion: {zone_name}", bbox, conf)
                
                # Loitering Logic
                frames_in_zone = frame_num - self.state[track_id][zone_name]
                time_in_zone = frames_in_zone / self.fps
                
                if time_in_zone >= zone['loiter_thresh']:
                    loiter_key = (track_id, zone_name)
                    if loiter_key not in self.logged_loiter:
                        logger.log(frame_num, track_id, f"Loitering: {zone_name}", bbox, conf)
                        self.logged_loiter.add(loiter_key)
            else:
                # Reset state if person leaves the zone
                if zone_name in self.state[track_id]:
                    del self.state[track_id][zone_name]
                    loiter_key = (track_id, zone_name)
                    if loiter_key in self.logged_loiter:
                        self.logged_loiter.remove(loiter_key)
                        
    def draw_zones(self, frame):
        for zone in self.zones:
            cv2.polylines(frame, [zone['pts']], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, zone['name'], tuple(zone['pts'][0][0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

def run_pipeline(video_path, config_path, output_video_path, output_log_path):
    # Load YOLOv8 model (nano version for speed, can swap for 'yolov8m.pt' for accuracy)
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    zone_manager = ZoneManager(config_path)
    # override config fps with actual video fps for accurate loitering time
    zone_manager.fps = fps if fps > 0 else 30 
    logger = EventLogger()
    
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        
        # Run YOLOv8 tracking, class 0 is 'person'
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        frame = zone_manager.draw_zones(frame)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                # Calculate bottom center point of bounding box for zone interaction
                bottom_center = (int((x1 + x2) / 2), y2)
                
                # Process logic
                zone_manager.process_point(frame_num, track_id, bottom_center, [int(x) for x in box], conf, logger)
                
                # Draw annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, bottom_center, 4, (255, 0, 0), -1)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        
        # Optional: uncomment to view processing live (may slow down CLI script)
        # cv2.imshow("Surveillance Pipeline", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save structured event logs
    logger.save(output_log_path)
    print(f"Processing complete. Video saved to {output_video_path}. Logs saved to {output_log_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Surveillance Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, required=True, help="Path to zone config JSON")
    parser.add_argument("--output_vid", type=str, default="output.mp4", help="Path to save annotated video")
    parser.add_argument("--output_log", type=str, default="events.json", help="Path to save event JSON")
    
    args = parser.parse_args()
    
    run_pipeline(args.input, args.config, args.output_vid, args.output_log)