import cv2
from ultralytics import YOLO
import numpy as np



class TrackState:
    def __init__(self, n_confirm=4, m_persist=20):
        self.n_confirm = n_confirm
        self.m_persist = m_persist
        self.tracks = {}

    def update(self, seen_ids):
        for track_id in list(self.tracks):
            t = self.tracks[track_id]
            if track_id in seen_ids:
                t['seen_count'] += 1
                t['missed_count'] = 0
                if t['seen_count'] >= self.n_confirm:
                    t['confirmed'] = True
            else:
                t['missed_count'] += 1

            if t['missed_count'] > self.m_persist:
                del self.tracks[track_id]
        for track_id in seen_ids:
            if track_id not in self.tracks:
                self.tracks[track_id] = {'seen_count': 1, 'missed_count': 0, 'confirmed': False}

    def is_visible(self, track_id):
        t = self.tracks.get(track_id)
        if t is None:
            return False
        if t['confirmed'] and t['missed_count'] <= self.m_persist:
            return True
        return False
        
class BoxSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_box = {}

    def smooth(self, box, track_id):
        box = np.asarray(box, dtype=np.float32)
        if track_id not in self.prev_box:
            self.prev_box[track_id] = box
            return box
        else:
            self.prev_box[track_id] = self.alpha * box + (1 - self.alpha) * self.prev_box[track_id]
            return self.prev_box[track_id]
    def drop(self, track_id):
        self.prev_box.pop(track_id, None)
            
       

def run(
    model_path,
    video_path,
    output_path,
    conf=0.25,
    imgsz=640,
    alpha=0.4,
    n_confirm=3,
    m_persist=15,
    draw_mask=True,
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video acilmadi.")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    state = TrackState(n_confirm=n_confirm, m_persist=m_persist)
    smoother = BoxSmoother(alpha=alpha)


    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, imgsz=imgsz)
        r = results[0]
        

        dets ={}
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            polys = r.masks.xy if (draw_mask and r.masks is not None) else None

            for i, track_id in enumerate(ids):
                poly = polys[i] if polys is not None else None
                dets[track_id] = (xyxy[i], poly)
 
        state.update(dets.keys())
 
        for track_id in list(state.tracks.keys()):
            if not state.is_visible(track_id):
                continue
 
            det = dets.get(track_id)
            if det is None:
             
                if track_id in smoother.prev_box:
                    box = smoother.prev_box[track_id]
                    mask = None
                else:
                    continue
            else:
                raw_box, mask = det
                box = smoother.smooth(raw_box,track_id)
 
            x1, y1, x2, y2 = map(int, box)
 
            if draw_mask and mask is not None and len(mask) > 0:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [mask.astype(np.int32)], (0, 200, 0))
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"pothole {track_id}", (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
 
        for track_id in list(smoother.prev_box.keys()):
            if track_id not in state.tracks:
                smoother.drop(track_id)
 
        out.write(frame)
 
    cap.release()
    out.release()
 

if __name__ == "__main__":
    run(
        model_path="best.pt",
        video_path="video.mp4",
        output_path="output.mp4",
        conf=0.25,
        imgsz=640,
        alpha=0.4,
        n_confirm=3,
        m_persist=15,
        draw_mask=True,
    )