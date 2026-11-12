"""Transient, in-memory frame relay for camera feeds.

Frames arrive from the browser (a still JPEG grabbed from a local video file
or webcam), are held here only long enough for the live-cameras page to pick
them up, and are then overwritten. Nothing is written to the database or to
disk, and no model changes are involved.

A feed is considered *running* while frames keep arriving. If the producer
stops (file ended, tab closed, webcam revoked) the entry goes stale and the
feed reports as *offline* without anybody having to write that status down.
"""

import threading
import time

# A feed is "running" only while frames are still arriving within this window.
STALE_AFTER_SECONDS = 4.0

# Guards against a runaway producer filling memory.
MAX_FRAME_BYTES = 2 * 1024 * 1024

# Cameras with no recent frames are dropped entirely after this long.
PURGE_AFTER_SECONDS = 120.0


class FeedRegistry:
    def __init__(self, stale_after=STALE_AFTER_SECONDS, purge_after=PURGE_AFTER_SECONDS):
        self._lock = threading.Lock()
        self._feeds = {}
        self._stale_after = stale_after
        self._purge_after = purge_after

    def publish(self, camera_id, frame, kind=''):
        """Store the newest frame for a camera and return its sequence number."""
        if not camera_id or not frame:
            return 0
        with self._lock:
            entry = self._feeds.get(camera_id)
            seq = (entry['seq'] + 1) if entry else 1
            if entry is None:
                entry = {'result_frame': None}
            entry.update({
                'raw_frame': bytes(frame),
                'seq': seq,
                'ts': time.monotonic(),
                'kind': kind or entry.get('kind', ''),
            })
            self._feeds[camera_id] = entry
            return seq

    def publish_result(self, camera_id, frame):
        """Attach a completed engine result without replacing the live input."""
        if not camera_id or not frame:
            return False
        with self._lock:
            entry = self._feeds.get(camera_id)
            if not entry:
                return False
            entry['result_frame'] = bytes(frame)
            return True

    def latest(self, camera_id):
        """Return the newest frame for a camera, or None if nothing is buffered."""
        with self._lock:
            entry = self._feeds.get(camera_id)
            return (entry.get('result_frame') or entry.get('raw_frame')) if entry else None

    def status(self, camera_id):
        with self._lock:
            entry = self._feeds.get(camera_id)
            if not entry:
                return {'running': False, 'seq': 0, 'age': None, 'kind': ''}
            age = time.monotonic() - entry['ts']
            return {
                'running': age <= self._stale_after,
                'seq': entry['seq'],
                'age': round(age, 2),
                'kind': entry['kind'],
            }

    def status_for(self, camera_ids):
        """Statuses for the given cameras, in the order requested."""
        return {camera_id: self.status(camera_id) for camera_id in camera_ids}

    def drop(self, camera_id):
        with self._lock:
            return self._feeds.pop(camera_id, None) is not None

    def purge(self):
        """Forget feeds that have been silent long enough to be irrelevant."""
        now = time.monotonic()
        with self._lock:
            dead = [
                camera_id
                for camera_id, entry in self._feeds.items()
                if now - entry['ts'] > self._purge_after
            ]
            for camera_id in dead:
                del self._feeds[camera_id]


registry = FeedRegistry()
