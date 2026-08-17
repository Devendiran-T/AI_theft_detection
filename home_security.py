import cv2
from ultralytics import YOLO
import requests
import datetime
import time
import os
import json
import threading
from collections import deque

# ============================================================
#  CONFIGURATION  — fill in your Telegram details below
# ============================================================
BOT_TOKEN = "8587553373:AAFBG40llI9689Qf9cI8eg2eGPSNeblNKBc"
CHAT_ID = "1510892667"

# Tuning knobs
COOLDOWN_SEC         = 60         # gap between photo bursts (1 min)
SUSTAINED_MIN        = 3          # minutes before "still here" alert fires
MULTI_PERSON_ALERT   = True       # alert when 2+ people detected at once
MIN_CONFIDENCE       = 0.55       # ignore detections below this confidence
SAVE_SNAPSHOTS       = False       # snapshot saving disabled; photos are sent via Telegram only
SNAPSHOT_DIR         = "snapshots"
LOG_FILE             = "detections.log"
ENABLE_ZONE          = False      # only alert when person is inside the zone below
ALERT_ZONE           = (100, 100, 540, 380)   # (x1, y1, x2, y2) in pixels
# ============================================================

# ── helpers ──────────────────────────────────────────────────

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def stamp_frame(frame, label=""):
    """Burn timestamp + optional label onto a copy of the frame."""
    out = frame.copy()
    ts  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out, ts,    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(out, label, (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,200,255), 2)
    return out

def draw_boxes(frame, results, model):
    """Draw bounding boxes for every person detected."""
    out = frame.copy()
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "person" and float(box.conf[0]) >= MIN_CONFIDENCE:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(out, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(out, f"Person {conf:.0%}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
    return out

def is_in_zone(box):
    """Check whether a bounding box overlaps the alert zone."""
    zx1,zy1,zx2,zy2 = ALERT_ZONE
    bx1,by1,bx2,by2 = map(int, box.xyxy[0])
    return bx1 < zx2 and bx2 > zx1 and by1 < zy2 and by2 > zy1

def is_dark(frame, threshold=40):
    """Return True if the frame is too dark (night / low light)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold

def send_telegram_message(text):
    """Send a plain text message → triggers one phone vibration."""
    url = f"https://api.telegram.org/bot8587553373:AAFBG40llI9689Qf9cI8eg2eGPSNeblNKBc/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        log(f"Message send failed: {e}")

def double_vibrate(message):
    """
    Send two rapid messages so the phone vibrates twice.
    Telegram triggers a vibration for each notification.
    """
    send_telegram_message(message)
    time.sleep(0.5)
    send_telegram_message("🔔 (alert confirmed)")

def send_telegram_photos(frames, caption=""):
    """Send up to 3 frames as separate photo messages."""
    url = f"https://api.telegram.org/bot8587553373:AAFBG40llI9689Qf9cI8eg2eGPSNeblNKBc/sendPhoto"
    for idx, frame in enumerate(frames[:3]):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        files = {"photo": (f"alert_{idx+1}.jpg", buf.tobytes(), "image/jpeg")}
        data  = {"chat_id": CHAT_ID, "caption": caption if idx == 0 else ""}
        try:
            requests.post(url, data=data, files=files, timeout=15)
        except Exception as e:
            log(f"Photo {idx+1} send failed: {e}")

def save_snapshot(frame, tag=""):
    """Save a frame to disk for local review."""
    if not SAVE_SNAPSHOTS:
        return
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"{ts}_{tag}.jpg")
    cv2.imwrite(path, frame)

def capture_burst(cap, base_frame, n=2):
    """Return base_frame + (n-1) freshly captured frames."""
    frames = [base_frame.copy()]
    for _ in range(n - 1):
        ok, f = cap.read()
        if ok:
            frames.append(f.copy())
    return frames

def alert_async(frames, message, caption, vibrate=False):
    """
    Run Telegram send in a background thread.
    vibrate=True  -> double vibrate + 3 photos  (first detection)
    vibrate=False -> single message + 3 photos  (1-min follow-up bursts)
    """
    def _send():
        if vibrate:
            double_vibrate(message)
        else:
            send_telegram_message(message)
        send_telegram_photos(frames, caption=caption)
    threading.Thread(target=_send, daemon=True).start()

# ── daily summary ─────────────────────────────────────────────

daily_stats = {"detections": 0, "multi_person_events": 0, "sustained_alerts": 0}
last_summary_day = datetime.date.today()

def maybe_send_daily_summary():
    global last_summary_day
    today = datetime.date.today()
    if today != last_summary_day:
        msg = (
            f"📊 Daily Summary — {last_summary_day}\n"
            f"  Person detections : {daily_stats['detections']}\n"
            f"  Multi-person events: {daily_stats['multi_person_events']}\n"
            f"  Sustained (3-min) alerts: {daily_stats['sustained_alerts']}"
        )
        send_telegram_message(msg)
        log(f"Daily summary sent for {last_summary_day}")
        daily_stats["detections"] = 0
        daily_stats["multi_person_events"] = 0
        daily_stats["sustained_alerts"] = 0
        last_summary_day = today

# ── FPS counter ───────────────────────────────────────────────

fps_buf   = deque(maxlen=30)
prev_time = time.time()

def update_fps():
    global prev_time
    now = time.time()
    fps_buf.append(1.0 / max(now - prev_time, 1e-9))
    prev_time = now
    return sum(fps_buf) / len(fps_buf)

# ── main ──────────────────────────────────────────────────────

def main():
    log("Loading YOLOv8 model…")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log("ERROR: Could not open camera.")
        return

    log("Monitoring started. Press ESC to quit.")

    last_alert_time     = 0          # 0 = never alerted (used to detect first detection)
    no_person_since = None  # timestamp when no person detected continuously
    person_first_seen   = None       # timestamp when continuous presence began
    last_person_count   = 0          # for multi-person tracking

    while True:
        ret, frame = cap.read()
        if not ret:
            log("Camera read failed — attempting reconnection…")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            continue

        # ── night detection ──────────────────────────────────
        night_mode = is_dark(frame)
        if night_mode:
            # Brighten for YOLO (don't show the brightened frame)
            proc_frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=30)
        else:
            proc_frame = frame

        # ── YOLO inference ───────────────────────────────────
        results = model(proc_frame, verbose=False)

        # Count persons passing confidence & zone filters
        person_boxes = []
        for r in results:
            for box in r.boxes:
                if (model.names[int(box.cls[0])] == "person"
                        and float(box.conf[0]) >= MIN_CONFIDENCE):
                    if ENABLE_ZONE and not is_in_zone(box):
                        continue
                    person_boxes.append(box)

        person_count = len(person_boxes)
        now          = time.time()

        # ── presence tracking ────────────────────────────────
        if person_count > 0:
            if person_first_seen is None:
                person_first_seen = now
            no_person_since = None
        else:
            # No person detected in this frame
            if no_person_since is None:
                no_person_since = now
            elif now - no_person_since >= 2:
                # Person has been absent for >=2 seconds, reset session state
                person_first_seen = None
                last_alert_time = 0
                no_person_since = None
                log("Person absent for 2 seconds, session reset")

        # ── photo burst every 60 seconds while person is present ─
        if person_count > 0 and (now - last_alert_time) >= COOLDOWN_SEC:
            # Determine if this is the first alert of the session
            is_first = (last_alert_time == 0)
            label_base = "INTRUDER DETECTED"
            if night_mode:
                label_base += " (low-light)"

            # Use burst number based on elapsed time since first seen
            burst_num = int((now - person_first_seen) / COOLDOWN_SEC) + 1
            msg = f"{label_base} | Burst #{burst_num} | {datetime.datetime.now().strftime('%H:%M:%S')}"
            log(msg)
            daily_stats["detections"] += 1

            # Capture exactly 2 frames (base + one extra)
            burst = capture_burst(cap, frame, n=2)
            stamped = [stamp_frame(draw_boxes(f, results, model), f"Burst #{burst_num}") for f in burst]
            for f in stamped:
                save_snapshot(f, f"burst{burst_num}")

            # Send alert (vibrate on first detection only)
            alert_async(stamped, msg, caption=msg, vibrate=False)
            last_alert_time = now

        # ── daily summary check ───────────────────────────────
        maybe_send_daily_summary()

        # ── live display ─────────────────────────────────────
        fps     = update_fps()
        display = draw_boxes(frame, results, model)

        # Status overlay
        status = (f"FPS: {fps:.1f}  |  Persons: {person_count}"
                  + ("  [NIGHT]" if night_mode else ""))
        if person_first_seen:
            elapsed = int(now - person_first_seen)
            status += f"  |  Presence: {elapsed}s"
        cv2.putText(display, status, (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        # Alert zone overlay
        if ENABLE_ZONE:
            zx1,zy1,zx2,zy2 = ALERT_ZONE
            cv2.rectangle(display, (zx1,zy1), (zx2,zy2), (255,165,0), 2)
            cv2.putText(display, "ALERT ZONE", (zx1, zy1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1)

        cv2.imshow("🔒 Home Security AI", display)
        last_person_count = person_count

        if cv2.waitKey(1) == 27:   # ESC
            break

    log("Monitoring stopped.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()