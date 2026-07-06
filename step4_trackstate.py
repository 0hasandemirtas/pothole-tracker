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