# face_auth.py
# Camera handling, face enrollment, face recognition, and auth selection.

import os
import sys
import platform
import re
import numpy as np

from config import (
    STRICT_AUTH,
    AUTO_RECOG_ON_START,
    CAMERA_INDEX,
    PREFER_EXTERNAL,
    ENROLL_CAMERA_INDEX,
    RECOG_CAMERA_INDEX,
    FACE_MIN_SIZE,
    FACE_BLUR_THRESH,
    FACE_MAX_BLUR,
    FACE_DIST_THRESH,
)

# Optional OpenCV
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False


# -------- camera backend helpers --------

if HAS_CV2:
    # helps avoid sensor conflicts on mac
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")


def _get_backend():
    """
    Pick correct cv2 backend depending on OS.
    """
    syst = platform.system()
    if syst == "Windows":
        return cv2.CAP_DSHOW if HAS_CV2 else 0
    if syst == "Darwin":
        return cv2.CAP_AVFOUNDATION if HAS_CV2 else 0
    if syst == "Linux":
        return cv2.CAP_V4L2 if HAS_CV2 else 0
    return 0


def _open_camera(max_indices=3, for_purpose=None):
    """
    Try to open a camera device with consistent index logic.

    Priority order:
    1. If ENROLL_CAMERA_INDEX is set and for_purpose == "enroll", force it.
    2. If RECOG_CAMERA_INDEX is set and for_purpose == "recognize", force it.
    3. If CAMERA_INDEX >= 0 is set, force that.
    4. Otherwise auto-pick between [1,0,2,...] or [0,1,2,...] depending on PREFER_EXTERNAL.

    We *do not* silently fall back if the user forced an index.
    """
    if not HAS_CV2:
        return None, None, None

    backend = _get_backend()

    # Hard requirement for enrollment camera
    if for_purpose == "enroll" and ENROLL_CAMERA_INDEX is not None:
        try_idx = int(ENROLL_CAMERA_INDEX)
        cap = cv2.VideoCapture(try_idx, backend)
        if cap.isOpened():
            bname = getattr(cap, "getBackendName", lambda: "unknown")()
            print(f"(Using ENROLL_CAMERA_INDEX {try_idx} - backend={bname})")
            return cap, try_idx, bname
        cap.release()
        print(f"(FATAL: ENROLL_CAMERA_INDEX {try_idx} not available. No fallback.)")
        return None, None, None

    # Hard requirement for recognition camera
    if for_purpose == "recognize" and RECOG_CAMERA_INDEX is not None:
        try_idx = int(RECOG_CAMERA_INDEX)
        cap = cv2.VideoCapture(try_idx, backend)
        if cap.isOpened():
            bname = getattr(cap, "getBackendName", lambda: "unknown")()
            print(f"(Using RECOG_CAMERA_INDEX {try_idx} - backend={bname})")
            return cap, try_idx, bname
        cap.release()
        print(f"(FATAL: RECOG_CAMERA_INDEX {try_idx} not available. No fallback.)")
        return None, None, None

    # Forced global CAMERA_INDEX
    if CAMERA_INDEX >= 0:
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if cap.isOpened():
            bname = getattr(cap, "getBackendName", lambda: "unknown")()
            print(f"(Using forced CAMERA_INDEX {CAMERA_INDEX} - backend={bname})")
            return cap, CAMERA_INDEX, bname
        cap.release()
        print(f"(FATAL: forced CAMERA_INDEX {CAMERA_INDEX} not available. No fallback.)")
        return None, None, None

    # Otherwise, auto logic
    if PREFER_EXTERNAL:
        preferred_order = [1, 0] + [i for i in range(2, max_indices + 1)]
    else:
        preferred_order = [0] + [i for i in range(1, max_indices + 1)]

    tried = set()
    for idx in preferred_order:
        if idx in tried:
            continue
        tried.add(idx)
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            bname = getattr(cap, "getBackendName", lambda: "unknown")()
            print(f"(Auto-selected camera index {idx} - backend={bname})")
            return cap, idx, bname
        cap.release()

    # Last sweep: brute force
    for idx in range(max_indices + 1):
        if idx in tried:
            continue
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            bname = getattr(cap, "getBackendName", lambda: "unknown")()
            print(f"(Fallback camera index {idx} - backend={bname})")
            return cap, idx, bname
        cap.release()

    return None, None, None


def _find_cascade(fname="haarcascade_frontalface_alt.xml"):
    """
    Try to locate the Haar cascade file for face detection.
    """
    if not HAS_CV2:
        return None

    data_obj = getattr(cv2, "data", None)
    data_dir = getattr(data_obj, "haarcascades", None) if data_obj else None
    if isinstance(data_dir, str):
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            return p

    roots = [
        os.path.dirname(getattr(cv2, "__file__", "")),
        sys.prefix,
        "/usr/share/opencv4",
        "/usr/local/share/opencv4",
        "/usr/share/opencv",
        "/usr/local/share/opencv",
    ]
    subdirs = [
        "data/haarcascades",
        "haarcascades",
        "share/opencv4/haarcascades",
        "",
    ]

    for r in roots:
        for s in subdirs:
            p = os.path.join(r, s, fname)
            if os.path.exists(p):
                return p

    # fallback to local path
    if os.path.exists(fname):
        return os.path.abspath(fname)

    return None


# -------- enrollment (capture_profile) --------

def capture_profile(name: str, *, keep_every=10, max_frames=1000) -> bool:
    """
    Capture face samples for a new identity:
      - grabs frames from camera
      - detects largest face
      - quality filter (not too blurry, not blown out)
      - resizes face ROI to 100x100
      - saves a matrix of flattened faces to face_dataset/<Name>.npy
    """
    if not HAS_CV2:
        print("(OpenCV not available)")
        return False

    cap, idx, bname = _open_camera(for_purpose="enroll")
    if cap is None:
        print("Could not open any camera. Close Zoom/Teams/etc. and check permissions.")
        return False

    print(f"Camera opened at index {idx} (backend={bname}). Press 'q' to quit.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cascade_path = _find_cascade()
    if not cascade_path:
        print("Haar cascade not found. Put haarcascade_frontalface_alt.xml in this folder or install opencv-data.")
        cap.release()
        return False

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Failed to load cascade at: {cascade_path}")
        cap.release()
        return False

    ds_dir = "face_dataset"
    os.makedirs(ds_dir, exist_ok=True)

    samples = []
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # allow quitting even if frame read fails
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces):
            # take largest face
            x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]

            if w >= FACE_MIN_SIZE and h >= FACE_MIN_SIZE:
                off = 5
                y1, y2 = max(0, y-off), min(frame.shape[0], y+h+off)
                x1, x2 = max(0, x-off), min(frame.shape[1], x+w+off)

                roi = gray[y1:y2, x1:x2]
                if roi.size:
                    roi100 = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)

                    # "sharpness" check
                    fm = cv2.Laplacian(roi100, cv2.CV_64F).var()
                    if FACE_BLUR_THRESH <= fm <= FACE_MAX_BLUR:
                        frame_counter += 1
                        if frame_counter % keep_every == 0:
                            samples.append(roi100)
                            cv2.imshow("Sample (100x100)", roi100)
                            print(f"Saved sample #{len(samples)} (sharpness={fm:.1f})")
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    elif fm < FACE_BLUR_THRESH:
                        # too blurry
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                        cv2.putText(frame, f"Too blurry ({fm:.0f})", (x,y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0,255,255), 1, cv2.LINE_AA)
                    else:
                        # too bright / blown out
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,165,255), 2)
                        cv2.putText(frame, f"Too bright ({fm:.0f})", (x,y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0,165,255), 1, cv2.LINE_AA)

        # HUD text
        cv2.putText(frame, "Press 'q' to finish",
                    (10,22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Enroll Face", frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or frame_counter >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        X = np.asarray(samples, dtype=np.uint8).reshape(len(samples), -1)
        out_path = os.path.join("face_dataset", f"{name}.npy")
        np.save(out_path, X)
        print("Saved:", os.path.abspath(out_path))
        return True

    print("No samples captured.")
    return False


# -------- recognition bank helpers --------

def _load_face_bank():
    """
    Returns:
      train -> ndarray [N, D+1] where last column is label ID
      names -> {class_id: name}
    or (None, {}) if not available.
    """
    if not (HAS_CV2 and os.path.isdir("face_dataset")):
        return None, {}

    feats = []
    labels = []
    names = {}
    cid = 0

    for fn in os.listdir("face_dataset"):
        if fn.endswith(".npy"):
            full = os.path.join("face_dataset", fn)
            X = np.load(full)
            if X.ndim != 2:
                continue
            names[cid] = fn[:-4]  # strip ".npy"
            feats.append(X)
            labels.append(np.full((X.shape[0],), cid, dtype=np.int32))
            cid += 1

    if not feats:
        return None, {}

    X_all = np.concatenate(feats, axis=0)
    y_all = np.concatenate(labels, axis=0).reshape(-1, 1)
    train = np.concatenate([X_all, y_all], axis=1)
    return train, names


def _euclid_dist(v1, v2):
    diff = (v1 - v2)
    return np.sqrt((diff * diff).sum())


def _knn(train, test_vec, k=5, threshold=FACE_DIST_THRESH):
    """
    Tiny kNN with rejection.
    Returns class_id or -1 if best match is still too far.
    """
    dists = []
    for i in range(train.shape[0]):
        dist = _euclid_dist(test_vec, train[i, :-1])
        lbl = train[i, -1]
        dists.append((dist, lbl))

    dists.sort(key=lambda t: t[0])

    # reject if best match is too far visually
    if dists[0][0] > threshold:
        return -1

    top = [lbl for _, lbl in dists[:k]]
    vals, counts = np.unique(top, return_counts=True)
    return int(vals[np.argmax(counts)])


def recognize_quick(*, timeout_frames=250, need_votes=5, k=5):
    """
    Attempt to ID a known face by watching camera frames briefly.
    If SAME predicted name hits need_votes votes, return that name.
    Otherwise return None.
    """
    if not HAS_CV2:
        return None

    train, names = _load_face_bank()
    if train is None:
        return None

    cap, idx, bname = _open_camera(for_purpose="recognize")
    if cap is None:
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cascade_path = _find_cascade()
    if not cascade_path:
        cap.release()
        return None

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        cap.release()
        return None

    votes = {}
    frames = 0

    while frames < timeout_frames:
        frames += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < FACE_MIN_SIZE or h < FACE_MIN_SIZE:
                continue

            off = 5
            y1, y2 = max(0,y-off), min(frame.shape[0], y+h+off)
            x1, x2 = max(0,x-off), min(frame.shape[1], x+w+off)
            roi = gray[y1:y2, x1:x2]
            if not roi.size:
                continue

            roi100 = cv2.resize(roi, (100,100), interpolation=cv2.INTER_AREA)

            fm = cv2.Laplacian(roi100, cv2.CV_64F).var()
            if not (FACE_BLUR_THRESH <= fm <= FACE_MAX_BLUR):
                continue

            pred_id = _knn(train, roi100.reshape(-1), k=k, threshold=FACE_DIST_THRESH)
            if pred_id == -1:
                continue

            name = names.get(pred_id)
            if not name:
                continue

            votes[name] = votes.get(name, 0) + 1
            if votes[name] >= need_votes:
                cap.release()
                cv2.destroyAllWindows()
                return name

    cap.release()
    cv2.destroyAllWindows()
    return None


# -------- high-level identity chooser --------

def choose_identity_from_faces_or_fallback(profiles: dict, *, try_camera=True):
    """
    Returns (identity, authed_bool)

    What we want in STRICT_AUTH mode:
    - Try to recognize face if AUTO_RECOG_ON_START and camera allowed.
    - If we get a match that looks like a real user profile (not Guest), return that user.
    - Otherwise return LOCKED (NOT Guest).

    In non-strict mode:
    - Try camera match; if found, return that.
    - Else fall back to last identity.
    - Else fall back to Guest.
    """
    who = None

    if try_camera and AUTO_RECOG_ON_START:
        who = recognize_quick(timeout_frames=250, need_votes=5, k=5)

    if STRICT_AUTH:
        # strict kiosk mode:
        if who and who in profiles and who.lower() != "guest":
            return who, True
        # no known face => locked
        return "LOCKED", False

    # non-strict mode:
    if who and who in profiles:
        return who, True

    last_id = profiles.get("_last_identity")
    if last_id and last_id in profiles:
        return last_id, True

    # final fallback without strict auth is Guest
    return "Guest", False
