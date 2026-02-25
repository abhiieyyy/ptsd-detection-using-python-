"""
============================================================
  camera_detect.py   (v5 — FIXED 31-feature match)

  ROOT CAUSE FIX:
      Old version extracted 7 features → model expected 31.
      This version extracts EXACTLY the same 31 columns
      that facial_dataset.csv and train_model.py use.

  HOW TO USE:
      python3 camera_detect.py
  CONTROLS:
      Q = quit    R = reset buffer    S = snapshot
============================================================
"""

import cv2, pickle, os, sys, math, ssl, urllib.request
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("pip3 install mediapipe"); sys.exit(1)

# ── Download landmarker model ────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading face_landmarker.task (~30 MB) ...")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(MODEL_URL, context=ctx) as r, \
         open(MODEL_PATH, "wb") as f:
        total = int(r.getheader("Content-Length", 0))
        done  = 0
        while True:
            chunk = r.read(8192)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r   {done*100//total}%", end="", flush=True)
    print(f"\nSaved {MODEL_PATH}")


# ═══════════════════════════════════════════════════════════════
#  31 FEATURE COLUMNS — must match facial_dataset.csv exactly
# ═══════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "AU1_intensity","AU4_intensity","AU5_intensity","AU6_intensity",
    "AU7_intensity","AU9_intensity","AU10_intensity","AU12_intensity",
    "AU15_intensity","AU17_intensity","AU20_intensity","AU23_intensity","AU24_intensity",
    "mouth_width_ratio","mouth_height_ratio",
    "eye_openness_left","eye_openness_right",
    "brow_raise_left","brow_raise_right",
    "nose_tip_x_norm","nose_tip_y_norm",
    "AU4_std","AU12_std","mouth_movement_std","eye_movement_std",
    "eye_asymmetry","brow_asymmetry","mouth_asymmetry",
    "head_yaw","head_pitch","head_roll",
]
assert len(FEATURE_COLS) == 31

# ── Landmark indices ─────────────────────────────────────────
L_EYE  = [33, 160, 158, 133, 153, 144]
R_EYE  = [362, 385, 387, 263, 373, 380]
L_BROW = [70, 63, 105, 66, 107]
R_BROW = [336, 296, 334, 293, 300]
NOSE   = 1; CHIN = 152; FOREHEAD = 10
L_CHK  = 234; R_CHK = 454
POSE_I = [1, 152, 33, 263, 61, 291]

PTS_3D = np.array([
    (  0.0,   0.0,   0.0),(  0.0,-330.0, -65.0),
    (-225.0, 170.0,-135.0),( 225.0, 170.0,-135.0),
    (-150.0,-150.0,-125.0),( 150.0,-150.0,-125.0),
], dtype=np.float64)


# ── Geometry helpers ─────────────────────────────────────────
def xy(lms, i, w, h):   return (lms[i].x*w, lms[i].y*h)
def dd(a, b):            return math.dist(a, b)

def ear_ratio(lms, idx, w, h):
    p = [xy(lms,i,w,h) for i in idx]
    return (dd(p[1],p[5])+dd(p[2],p[4])) / (2.0*dd(p[0],p[3])+1e-6)

def mar_ratio(lms, w, h):
    return dd(xy(lms,13,w,h),xy(lms,14,w,h)) / (dd(xy(lms,61,w,h),xy(lms,291,w,h))+1e-6)

def brow_raise(lms, brow, eye, w, h):
    fh = abs(lms[FOREHEAD].y-lms[CHIN].y)*h+1e-6
    return float((np.mean([lms[i].y*h for i in eye])-np.mean([lms[i].y*h for i in brow]))/fh)

def face_asym(lms, li, ri, w, h):
    nx = lms[NOSE].x*w
    fw = abs(lms[L_CHK].x-lms[R_CHK].x)*w+1e-6
    return float(np.mean([abs(2*nx - lms[l].x*w - lms[r].x*w) for l,r in zip(li,ri)])/fw)

def head_pose(lms, w, h):
    img = np.array([(lms[i].x*w,lms[i].y*h) for i in POSE_I],dtype=np.float64)
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float64)
    ok,rv,_ = cv2.solvePnP(PTS_3D,img,cam,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0,0.0,0.0
    rm,_ = cv2.Rodrigues(rv)
    a,*_ = cv2.RQDecomp3x3(rm)
    return float(a[1]),float(a[0]),float(a[2])

def calc_aus(lms, w, h):
    fh = abs(lms[FOREHEAD].y-lms[CHIN].y)*h+1e-6
    fw = abs(lms[L_CHK].x-lms[R_CHK].x)*w+1e-6
    el = ear_ratio(lms,L_EYE,w,h)
    er = ear_ratio(lms,R_EYE,w,h)
    m  = mar_ratio(lms,w,h)
    aus = {}
    aus["AU1"]  = float(np.clip((lms[159].y*h-(lms[107].y+lms[336].y)/2*h)/fh*12,0,5))
    aus["AU4"]  = float(np.clip((1-abs(lms[107].x-lms[336].x)*w/fw)*6,0,5))
    aus["AU5"]  = float(np.clip(((el+er)/2-0.15)*18,0,5))
    aus["AU6"]  = float(np.clip((lms[13].y*h-lms[116].y*h)/fh*10,0,5))
    aus["AU7"]  = float(np.clip((0.38-(el+er)/2)*22,0,5))
    nw=abs(lms[49].x-lms[279].x)*w
    aus["AU9"]  = float(np.clip((nw/fw-0.17)*28,0,5))
    aus["AU10"] = float(np.clip((lms[2].y*h-lms[13].y*h)/fh*15,0,5))
    my=(lms[61].y+lms[291].y)/2*h
    aus["AU12"] = float(np.clip((lms[0].y*h-my)/fh*22,0,5))
    aus["AU15"] = float(np.clip((my-lms[0].y*h)/fh*22,0,5))
    aus["AU17"] = float(np.clip((lms[152].y*h-lms[17].y*h)/fh*12,0,5))
    aus["AU20"] = float(np.clip((dd(xy(lms,61,w,h),xy(lms,291,w,h))/fw-0.38)*12,0,5))
    aus["AU23"] = float(np.clip((0.08-m)*40,0,5))
    aus["AU24"] = float(np.clip((0.06-m)*45,0,5))
    return aus, el, er


# ═══════════════════════════════════════════════════════════════
#  EXTRACT ALL 31 FEATURES
# ═══════════════════════════════════════════════════════════════
def extract_features(lms, w, h, history):
    aus, el, er = calc_aus(lms, w, h)
    m           = mar_ratio(lms, w, h)
    yaw, pitch, roll = head_pose(lms, w, h)

    f = {}
    for au in ["AU1","AU4","AU5","AU6","AU7","AU9","AU10",
               "AU12","AU15","AU17","AU20","AU23","AU24"]:
        f[f"{au}_intensity"] = aus.get(au, 0.0)

    f["mouth_width_ratio"]  = abs(lms[61].x - lms[291].x)
    f["mouth_height_ratio"] = m
    f["eye_openness_left"]  = el
    f["eye_openness_right"] = er
    f["brow_raise_left"]    = brow_raise(lms, L_BROW, L_EYE, w, h)
    f["brow_raise_right"]   = brow_raise(lms, R_BROW, R_EYE, w, h)
    f["nose_tip_x_norm"]    = lms[NOSE].x
    f["nose_tip_y_norm"]    = lms[NOSE].y

    if len(history) >= 5:
        f["AU4_std"]            = float(np.std([hh["AU4_intensity"]      for hh in history]))
        f["AU12_std"]           = float(np.std([hh["AU12_intensity"]     for hh in history]))
        f["mouth_movement_std"] = float(np.std([hh["mouth_height_ratio"] for hh in history])*20)
        f["eye_movement_std"]   = float(np.std(
            [(hh["eye_openness_left"]+hh["eye_openness_right"])/2 for hh in history])*20)
    else:
        f["AU4_std"]=f["AU12_std"]=f["mouth_movement_std"]=f["eye_movement_std"]=0.3

    f["eye_asymmetry"]   = face_asym(lms, L_EYE[:2],  R_EYE[:2],  w, h)
    f["brow_asymmetry"]  = face_asym(lms, L_BROW[:2], R_BROW[:2], w, h)
    f["mouth_asymmetry"] = abs(lms[61].y - lms[291].y)
    f["head_yaw"]        = yaw
    f["head_pitch"]      = pitch
    f["head_roll"]       = roll

    return f, el, er, aus


def to_vec(f):
    return [float(f.get(k, 0.0)) for k in FEATURE_COLS]  # always 31


# ═══════════════════════════════════════════════════════════════
#  OVERLAY
# ═══════════════════════════════════════════════════════════════
PTSD_THRESHOLD = 6.5   # severity must be >= this to show PTSD warning

def sev_lbl(s):
    if s < PTSD_THRESHOLD: return "No PTSD Indicated"
    if s <= 7.5:           return "Moderate PTSD"
    if s <= 9.0:           return "High PTSD"
    return "Severe PTSD"

PTSD_C=(0,45,200); OK_C=(0,175,55); PEND_C=(0,150,220)

def draw_ui(frame, result, conf, sev, frames, needed, aus, el, er):
    h,w = frame.shape[:2]
    # Colour driven by severity threshold, not raw binary prediction
    ptsd_active = (result is not None) and (sev >= PTSD_THRESHOLD)
    col  = PEND_C if result is None else (PTSD_C if ptsd_active else OK_C)
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,130),col,-1)
    cv2.addWeighted(ov,0.50,frame,0.50,0,frame)

    if result is None:
        l1=f"COLLECTING — hold still  ({frames}/{needed} frames)"
        l2="Detecting facial features ..."
    elif sev >= PTSD_THRESHOLD:
        l1="WARNING: PTSD Indicators Detected"
        l2=f"Confidence: {conf:.1%}    Severity: {sev:.1f}/10  ({sev_lbl(sev)})"
    else:
        l1="No PTSD Indicators Detected"
        l2=f"Confidence: {conf:.1%}    Severity: {sev:.1f}/10  ({sev_lbl(sev)})"

    cv2.putText(frame,l1,(15,46),cv2.FONT_HERSHEY_DUPLEX,0.85,(255,255,255),2)
    cv2.putText(frame,l2,(15,82),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
    cv2.putText(frame,"Research prototype — NOT a clinical diagnosis",
                (15,112),cv2.FONT_HERSHEY_SIMPLEX,0.37,(210,210,255),1)

    px=w-225
    cv2.rectangle(frame,(px-5,0),(w,270),(22,22,22),-1)
    cv2.putText(frame,"Live Features",(px,22),cv2.FONT_HERSHEY_SIMPLEX,0.48,(170,170,170),1)
    for i,(lbl,val) in enumerate([
        ("EAR left",f"{el:.3f}"),("EAR right",f"{er:.3f}"),
        ("AU4 distress",f"{aus.get('AU4',0):.2f}"),
        ("AU6 cheek",f"{aus.get('AU6',0):.2f}"),
        ("AU12 smile",f"{aus.get('AU12',0):.2f}"),
        ("AU20 fear",f"{aus.get('AU20',0):.2f}"),
        ("AU23 tense",f"{aus.get('AU23',0):.2f}"),
    ]):
        y=50+i*28
        cv2.putText(frame,f"{lbl}:",(px,y),cv2.FONT_HERSHEY_SIMPLEX,0.37,(140,195,140),1)
        cv2.putText(frame,val,(px+140,y),cv2.FONT_HERSHEY_SIMPLEX,0.37,(255,255,100),1)

    by=h-32; bw=w-30
    cv2.rectangle(frame,(15,by),(bw,by+13),(50,50,50),-1)
    cv2.rectangle(frame,(15,by),(15+int((bw-15)*min(frames/max(needed,1),1.0)),by+13),col,-1)
    cv2.putText(frame,"[Q]uit  [R]eset  [S]napshot",
                (15,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.36,(180,180,180),1)
    return frame


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    # Load model — tries model.pkl first, then model/facial_bundle.pkl
    for path in ("model.pkl", os.path.join("model","facial_bundle.pkl")):
        if os.path.exists(path):
            bundle_path = path; break
    else:
        print("No model found. Run: python3 train_model.py"); sys.exit(1)

    with open(bundle_path,"rb") as fh:
        data = pickle.load(fh)

    scaler    = data.get("scaler")
    model     = data.get("model") or data.get("best_model")
    regressor = data.get("regressor")

    if None in (scaler, model, regressor):
        print("Bundle incomplete — re-run python3 train_model.py"); sys.exit(1)

    expected = scaler.n_features_in_
    if expected != 31:
        print(f"Scaler expects {expected} features but camera extracts 31.")
        print("Re-run: python3 generate_dataset.py && python3 train_model.py")
        sys.exit(1)

    print(f"Loaded model: {bundle_path}  ({expected} features confirmed)")

    download_model()

    # MediaPipe Tasks — VIDEO mode for real-time
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.50,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible.")
        print("Mac: System Preferences -> Privacy & Security -> Camera -> enable Terminal")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    NEEDED   = 75          # 5 sec × ~15 fps
    feat_buf = deque(maxlen=NEEDED)
    hist_buf = deque(maxlen=30)

    result=None; conf=0.0; sev=0.0; aus={}; el=er=0.0; ts=0
    os.makedirs("snapshots",exist_ok=True); snap_n=0

    print(f"\nCamera open — collecting {NEEDED} frames (~5 sec).")
    print("Q=quit  R=reset  S=snapshot\n")

    while True:
        ret,frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        h,w   = frame.shape[:2]
        ts   += 33

        rgb    = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        det    = detector.detect_for_video(mp_img,ts)

        if det.face_landmarks:
            lms = det.face_landmarks[0]
            for idx in L_EYE+R_EYE+L_BROW+R_BROW+[NOSE]:
                if idx < len(lms):
                    cv2.circle(frame,(int(lms[idx].x*w),int(lms[idx].y*h)),2,(0,255,180),-1)

            fd,el,er,aus = extract_features(lms,w,h,hist_buf)
            hist_buf.append(fd)
            feat_buf.append(to_vec(fd))  # 31 floats

            if len(feat_buf) >= NEEDED:
                avg = np.mean(list(feat_buf),axis=0).reshape(1,-1)  # (1,31)
                Xs  = scaler.transform(avg)
                pred = int(model.predict(Xs)[0])
                conf = (float(model.predict_proba(Xs)[0][pred])
                        if hasattr(model,"predict_proba") else 0.75)
                sev  = round(float(np.clip(regressor.predict(Xs)[0],0,10)),1)
                result = pred

        frame = draw_ui(frame,result,conf,sev,len(feat_buf),NEEDED,aus,el,er)
        cv2.imshow("PTSD Facial Detector",frame)

        key = cv2.waitKey(1)&0xFF
        if key in (ord("q"),27):
            break
        elif key==ord("r"):
            feat_buf.clear(); hist_buf.clear(); result=None; print("Reset.")
        elif key==ord("s"):
            fname=f"snapshots/snap_{snap_n:03d}.png"
            cv2.imwrite(fname,frame)
            lbl = "PTSD" if (result is not None and sev>=PTSD_THRESHOLD) else ("Pending" if result is None else "Non-PTSD")
            print(f"Snapshot: {fname}  {lbl}  conf={conf:.1%}")
            snap_n+=1

    cap.release(); cv2.destroyAllWindows(); detector.close()
    print("Done.")

if __name__=="__main__":
    main()