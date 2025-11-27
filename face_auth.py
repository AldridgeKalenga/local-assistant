# face_auth.py
# Camera handling, face enrollment, face recognition, and auth selection.

import os
import sys
import platform
import re
import warnings
import numpy as np

# Suppress deprecation warning from face_recognition_models (pkg_resources)
# This is a third-party library issue, not our code
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

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
    FACE_LBPH_CONFIDENCE,
    FACE_RECOGNITION_TOLERANCE,
    FACE_CAMERA_AUTO,
)

# Optional OpenCV
try:
    import cv2
    HAS_CV2 = True
    # Check if face module is available (opencv-contrib-python)
    try:
        # Try to access the face module
        _ = cv2.face
        # Try to create a recognizer to verify it works
        _test_recognizer = cv2.face.LBPHFaceRecognizer_create()
        del _test_recognizer
        HAS_CV2_FACE = True
    except (AttributeError, Exception):
        HAS_CV2_FACE = False
except Exception:
    cv2 = None
    HAS_CV2 = False
    HAS_CV2_FACE = False

# Optional face_recognition library (best option - uses dlib deep learning)
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except Exception:
    face_recognition = None
    HAS_FACE_RECOGNITION = False


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


def _configure_camera_for_face_recognition(cap, enable_auto=True):
    """
    Configure camera for face recognition.
    
    Args:
        enable_auto: If True, enable auto-exposure and auto white balance for recognition.
                     If False, use fixed settings (better for enrollment consistency).
    """
    if cap is None or not HAS_CV2:
        return
    
    if enable_auto:
        # Enable auto exposure and white balance for recognition
        # This allows the camera to adapt to different lighting conditions
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Auto white balance
    else:
        # Fixed settings for enrollment (more consistent samples)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Manual white balance
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4000)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
        cap.set(cv2.CAP_PROP_CONTRAST, 50)
        cap.set(cv2.CAP_PROP_SATURATION, 50)
    
    # Set frame width and height for consistency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


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
    _configure_camera_for_face_recognition(cap, enable_auto=False)  # Fixed settings for enrollment

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
                            # Normalize the face image for better recognition across lighting conditions
                            normalized_face = _normalize_face_image(roi100)
                            samples.append(normalized_face)
                            cv2.imshow("Sample (100x100)", normalized_face)
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
        
        # Also generate and save face encodings if face_recognition is available
        if HAS_FACE_RECOGNITION:
            try:
                encodings_list = []
                for sample in samples:
                    # Convert grayscale to RGB for face_recognition
                    face_rgb = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB) if HAS_CV2 else sample
                    face_encodings = face_recognition.face_encodings(face_rgb)
                    if face_encodings:
                        encodings_list.append(face_encodings[0])
                
                if encodings_list:
                    # Update the saved encodings file
                    encodings_path = os.path.join("face_dataset", "face_encodings.npy")
                    names_path = os.path.join("face_dataset", "face_encodings_names.json")
                    
                    # Load existing encodings
                    existing_encodings = []
                    existing_names = []
                    if os.path.exists(encodings_path) and os.path.exists(names_path):
                        try:
                            import json
                            existing_encodings = np.load(encodings_path).tolist()
                            with open(names_path, 'r') as f:
                                existing_names = json.load(f)
                            # Remove old encodings for this name
                            existing_encodings = [e for i, e in enumerate(existing_encodings) 
                                                if existing_names[i] != name]
                            existing_names = [n for n in existing_names if n != name]
                        except Exception:
                            pass
                    
                    # Add new encodings
                    existing_encodings.extend(encodings_list)
                    existing_names.extend([name] * len(encodings_list))
                    
                    # Save updated encodings
                    np.save(encodings_path, np.array(existing_encodings))
                    import json
                    with open(names_path, 'w') as f:
                        json.dump(existing_names, f)
                    print(f"(Also saved {len(encodings_list)} face encodings for faster recognition)")
            except Exception as e:
                print(f"(Warning: Could not save face encodings: {e})")
        
        return True

    print("No samples captured.")
    return False


# -------- recognition bank helpers --------

def _load_face_encodings():
    """
    Load face encodings using face_recognition library (best option).
    Returns:
      known_encodings -> list of 128-dim face encodings
      known_names -> list of names corresponding to encodings
    """
    if not HAS_FACE_RECOGNITION:
        return None, None
    
    if not os.path.isdir("face_dataset"):
        return None, None
    
    known_encodings = []
    known_names = []
    
    # Check for saved encodings first
    encodings_path = os.path.join("face_dataset", "face_encodings.npy")
    names_path = os.path.join("face_dataset", "face_encodings_names.json")
    
    if os.path.exists(encodings_path) and os.path.exists(names_path):
        try:
            import json
            known_encodings = np.load(encodings_path).tolist()
            with open(names_path, 'r') as f:
                known_names = json.load(f)
            if known_encodings and known_names:
                return known_encodings, known_names
        except Exception as e:
            print(f"(Warning: Could not load saved encodings: {e})")
    
    # Generate encodings from saved face images
    all_encodings = []
    all_names = []
    
    for fn in sorted(os.listdir("face_dataset")):
        if fn.endswith(".npy") and not fn.startswith("face_encodings") and fn != "lbph_model.yml":
            full = os.path.join("face_dataset", fn)
            X = np.load(full)
            if X.ndim != 2:
                continue
            name = fn[:-4]  # strip ".npy"
            
            # Convert each sample to RGB format for face_recognition
            # face_recognition expects RGB images
            for i in range(X.shape[0]):
                face_img = X[i].reshape(100, 100).astype(np.uint8)
                # Convert grayscale to RGB (face_recognition needs RGB)
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB) if HAS_CV2 else face_img
                
                try:
                    # Get face encoding (128-dimensional vector)
                    encodings = face_recognition.face_encodings(face_rgb)
                    if encodings:
                        all_encodings.append(encodings[0])
                        all_names.append(name)
                except Exception:
                    continue
    
    if not all_encodings:
        return None, None
    
    # Save encodings for faster loading next time
    try:
        np.save(encodings_path, np.array(all_encodings))
        import json
        with open(names_path, 'w') as f:
            json.dump(all_names, f)
    except Exception as e:
        print(f"(Warning: Could not save encodings: {e})")
    
    return all_encodings, all_names


def _load_face_bank():
    """
    Load face samples and train/load LBPH face recognizer.
    Returns:
      recognizer -> cv2.face.LBPHFaceRecognizer or None
      names -> {class_id: name}
    """
    if not HAS_CV2:
        return None, {}
    
    if not HAS_CV2_FACE:
        print("(Error: OpenCV face module not available. Install with: pip install opencv-contrib-python)")
        return None, {}
    
    # Check if we have a trained model saved
    model_path = os.path.join("face_dataset", "lbph_model.yml")
    names_path = os.path.join("face_dataset", "names.json")
    
    if os.path.exists(model_path) and os.path.exists(names_path):
        try:
            import json
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(model_path)
            with open(names_path, 'r') as f:
                names = json.load(f)
                # Convert string keys back to int
                names = {int(k): v for k, v in names.items()}
            return recognizer, names
        except Exception as e:
            print(f"(Warning: Could not load saved model: {e})")
    
    # Train new model from face samples
    if not os.path.isdir("face_dataset"):
        return None, {}
    
    faces = []
    labels = []
    names = {}
    cid = 0
    
    for fn in sorted(os.listdir("face_dataset")):
        if fn.endswith(".npy"):
            full = os.path.join("face_dataset", fn)
            X = np.load(full)
            if X.ndim != 2:
                continue
            name = fn[:-4]  # strip ".npy"
            names[cid] = name
            
            # Reshape each sample back to 100x100 for LBPH
            for i in range(X.shape[0]):
                face_img = X[i].reshape(100, 100)
                faces.append(face_img)
                labels.append(cid)
            
            cid += 1
    
    if not faces:
        return None, {}
    
    # Train LBPH recognizer
    # Parameters: radius=1, neighbors=8, grid_x=8, grid_y=8
    # LBPH is robust to lighting variations by design
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    
    faces_array = np.array(faces, dtype=np.uint8)
    labels_array = np.array(labels, dtype=np.int32)
    
    recognizer.train(faces_array, labels_array)
    
    # Save the trained model
    try:
        recognizer.write(model_path)
        import json
        with open(names_path, 'w') as f:
            json.dump(names, f)
    except Exception as e:
        print(f"(Warning: Could not save model: {e})")
    
    return recognizer, names


def _normalize_face_image(face_img):
    """
    Normalize a face image to be more robust to lighting variations.
    Applies histogram equalization to handle different lighting conditions.
    """
    if not HAS_CV2:
        return face_img
    
    # Apply histogram equalization to normalize lighting
    # This redistributes pixel intensities to use the full 0-255 range
    # making the image more consistent across different lighting conditions
    equalized = cv2.equalizeHist(face_img)
    
    return equalized


def recognize_quick(*, timeout_frames=250, need_votes=5, confidence_threshold=None):
    """
    Attempt to ID a known face by watching camera frames briefly.
    Uses face_recognition library (best) or falls back to LBPH.
    If SAME predicted name hits need_votes votes, return that name.
    Otherwise return None.
    
    Args:
        timeout_frames: Maximum frames to process
        need_votes: Number of consistent predictions needed
        confidence_threshold: LBPH confidence threshold (only used if face_recognition unavailable)
    """
    if not HAS_CV2:
        return None

    # Try face_recognition first (most robust)
    if HAS_FACE_RECOGNITION:
        return _recognize_with_face_recognition(timeout_frames, need_votes)
    
    # Fall back to LBPH
    if HAS_CV2_FACE:
        return _recognize_with_lbph(timeout_frames, need_votes, confidence_threshold)
    
    return None


def _recognize_with_face_recognition(timeout_frames=250, need_votes=5):
    """
    Recognition using face_recognition library (dlib-based, most robust).
    """
    known_encodings, known_names = _load_face_encodings()
    if not known_encodings or not known_names:
        return None

    cap, idx, bname = _open_camera(for_purpose="recognize")
    if cap is None:
        return None

    _configure_camera_for_face_recognition(cap, enable_auto=FACE_CAMERA_AUTO)

    votes = {}
    frames = 0

    while frames < timeout_frames:
        frames += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # or "cnn" for better accuracy but slower
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(
                known_encodings, 
                face_encoding, 
                tolerance=FACE_RECOGNITION_TOLERANCE
            )
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            # Find best match
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                # Check if the match is within tolerance
                if best_match_index is not None and bool(matches[best_match_index]):
                    name = known_names[best_match_index]
                    votes[name] = votes.get(name, 0) + 1
                    if votes[name] >= need_votes:
                        cap.release()
                        cv2.destroyAllWindows()
                        return name

    cap.release()
    cv2.destroyAllWindows()
    return None


def _recognize_with_lbph(timeout_frames=250, need_votes=5, confidence_threshold=None):
    """
    Recognition using LBPH (fallback if face_recognition unavailable).
    """
    if confidence_threshold is None:
        confidence_threshold = FACE_LBPH_CONFIDENCE

    recognizer, names = _load_face_bank()
    if recognizer is None:
        return None

    cap, idx, bname = _open_camera(for_purpose="recognize")
    if cap is None:
        return None

    _configure_camera_for_face_recognition(cap, enable_auto=FACE_CAMERA_AUTO)

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

            # Quality check
            fm = cv2.Laplacian(roi100, cv2.CV_64F).var()
            if not (FACE_BLUR_THRESH <= fm <= FACE_MAX_BLUR):
                continue

            # Normalize the face image
            normalized_face = _normalize_face_image(roi100)
            
            # Predict using LBPH recognizer
            try:
                label_id, confidence = recognizer.predict(normalized_face)
                
                # Reject if confidence is too high (bad match)
                if confidence > confidence_threshold:
                    continue
                
                name = names.get(label_id)
                if not name:
                    continue

                votes[name] = votes.get(name, 0) + 1
                if votes[name] >= need_votes:
                    cap.release()
                    cv2.destroyAllWindows()
                    return name
            except Exception:
                continue

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
        who = recognize_quick(timeout_frames=250, need_votes=5)

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
