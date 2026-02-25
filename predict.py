

"""
============================================================
  predict.py   (v4 — Fixed bundle keys)

  KEY FIX: train_model.py saves bundles with key "model"
  (not "best_model"), and no "best_name" key.
  This version reads those keys correctly.

  HOW TO USE:  python3 predict.py
  Commands at prompt:
      camera  = open live webcam detector
      quit    = exit
============================================================
"""

import pickle, os, sys, math, re, string
import numpy as np

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "yours","yourself","yourselves","he","him","his","himself","she",
    "her","hers","herself","it","its","itself","they","them","their",
    "theirs","themselves","what","which","who","whom","this","that",
    "these","those","am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","through",
    "during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","both",
    "each","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","than","too","very","s","t","can","will",
    "just","don","should","now","d","ll","m","o","re","ve","y","ain",
    "aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
}
KEEP = {"not","never","no","can't","won't","don't","didn't","isn't","wasn't"}

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("","", '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for w in text.split():
        if w in KEEP:
            tokens.append(w); continue
        wc = w.replace("'","").replace("-","")
        if wc not in STOPWORDS and len(wc) > 2:
            for sfx in ("ing","ness","ment","tion","tions","edly","ed","er","est"):
                if wc.endswith(sfx) and len(wc)-len(sfx) > 3:
                    wc = wc[:-len(sfx)]; break
            tokens.append(wc)
    return " ".join(tokens)


def load_bundle(path):
    if not os.path.exists(path):
        return None
    with open(path,"rb") as f:
        return pickle.load(f)

def get_model(b):
    """Handles both 'model' (new) and 'best_model' (old) key."""
    return b.get("model") or b.get("best_model")

def get_name(b):
    if "best_name" in b:
        return b["best_name"]
    m = get_model(b)
    return type(m).__name__ if m else "Unknown"

def get_conf(model, X, pred):
    if hasattr(model,"predict_proba"):
        return float(model.predict_proba(X)[0][pred])
    if hasattr(model,"decision_function"):
        s = model.decision_function(X)
        s = float(s[0] if hasattr(s,"__len__") else s)
        return 1/(1+math.exp(-abs(s)))
    return 0.70

def predict_text(b, raw):
    X    = b["vectorizer"].transform([preprocess(raw)])
    m    = get_model(b)
    pred = int(m.predict(X)[0])
    conf = get_conf(m, X, pred)
    sev  = round(float(np.clip(b["regressor"].predict(X)[0], 0, 10)), 2)
    return pred, conf, sev

def predict_structured(b, vec):
    X    = b["scaler"].transform([vec])
    m    = get_model(b)
    pred = int(m.predict(X)[0])
    conf = get_conf(m, X, pred)
    sev  = round(float(np.clip(b["regressor"].predict(X)[0], 0, 10)), 2)
    return pred, conf, sev


def sim_audio(ptsd=True):
    np.random.seed(None)
    f = []
    for j in range(1,14):
        f.append(np.random.normal(-18+j*1.2 if ptsd else -10+j*1.0, 4.0))
    for j in range(1,14):
        f.append(np.clip(np.random.normal(3.5 if ptsd else 5.0, 1.5), 0.5, 12))
    f += [
        np.random.normal(100 if ptsd else 135, 15),
        np.clip(np.random.normal(30 if ptsd else 20,10),2,60),
        np.clip(np.random.normal(70 if ptsd else 45,20),5,140),
        np.random.normal(0.012 if ptsd else 0.022, 0.004),
        np.clip(np.random.normal(0.018 if ptsd else 0.012,0.006),0.001,0.05),
        np.clip(np.random.normal(0.06  if ptsd else 0.08, 0.02),0.02,0.2),
        np.clip(np.random.normal(0.009 if ptsd else 0.004,0.004),0.0005,0.03),
        np.clip(np.random.normal(0.85  if ptsd else 0.45, 0.3),0.05,2.5),
        np.clip(np.random.normal(0.07  if ptsd else 0.04, 0.03),0.005,0.2),
        np.random.normal(6.5 if ptsd else 15.0, 3.0),
        np.clip(np.random.normal(2.8 if ptsd else 4.5,0.8),1,7),
        np.clip(np.random.normal(0.45 if ptsd else 0.22,0.1),0.05,0.8),
        np.random.normal(1800 if ptsd else 2600, 400),
        np.random.normal(3200 if ptsd else 4400, 600),
        np.clip(np.random.normal(0.045 if ptsd else 0.062,0.012),0.01,0.12),
    ]
    return f

def sim_facial(ptsd=True):
    np.random.seed(None)
    ap = {"AU1":1.8,"AU4":2.9,"AU5":1.5,"AU6":0.4,"AU7":1.6,"AU9":0.3,
          "AU10":0.5,"AU12":0.3,"AU15":1.2,"AU17":1.5,"AU20":2.2,"AU23":1.8,"AU24":2.1}
    an = {"AU1":1.0,"AU4":0.6,"AU5":0.4,"AU6":1.8,"AU7":0.7,"AU9":0.8,
          "AU10":0.9,"AU12":2.1,"AU15":0.3,"AU17":0.4,"AU20":0.3,"AU23":0.4,"AU24":0.3}
    d  = ap if ptsd else an
    f  = [np.clip(np.random.normal(d[k],0.5),0,5) for k in d]
    f += [
        np.clip(np.random.normal(0.32 if ptsd else 0.38,0.04),0.15,0.55),
        np.clip(np.random.normal(0.08 if ptsd else 0.12,0.03),0.01,0.25),
        np.clip(np.random.normal(0.28 if ptsd else 0.38,0.06),0.05,0.6),
        np.clip(np.random.normal(0.27 if ptsd else 0.37,0.06),0.05,0.6),
        np.clip(np.random.normal(0.35 if ptsd else 0.22,0.08),0.02,0.65),
        np.clip(np.random.normal(0.34 if ptsd else 0.21,0.08),0.02,0.65),
        np.random.normal(0.5,0.05),
        np.clip(np.random.normal(0.55 if ptsd else 0.52,0.04),0.3,0.75),
        np.clip(np.random.normal(0.8  if ptsd else 0.3, 0.2),0.05,2.0),
        np.clip(np.random.normal(0.25 if ptsd else 0.4, 0.15),0.02,1.0),
        np.clip(np.random.normal(2.1  if ptsd else 1.2, 0.5),0.3,4.0),
        np.clip(np.random.normal(1.8  if ptsd else 0.9, 0.4),0.2,3.5),
        np.clip(np.random.normal(0.06 if ptsd else 0.02,0.02),0,0.15),
        np.clip(np.random.normal(0.05 if ptsd else 0.015,0.02),0,0.12),
        np.clip(np.random.normal(0.04 if ptsd else 0.01,0.015),0,0.1),
        np.random.normal(0, 8 if ptsd else 5),
        np.random.normal(-5 if ptsd else -2, 6 if ptsd else 4),
        np.random.normal(0, 4 if ptsd else 3),
    ]
    return f


def sev_lbl(s):
    if s<=2: return "Minimal"
    if s<=4: return "Low"
    if s<=6: return "Moderate"
    if s<=8: return "High"
    return "Severe"

def print_result(mod, pred, conf, sev):
    icon  = "⚠️ " if pred==1 else "✅"
    label = "PTSD Indicators DETECTED" if pred==1 else "No PTSD Indicators"
    bar   = "█"*int(sev) + "░"*(10-int(sev))
    print(f"    {mod:14s} {icon} {label}")
    print(f"                    Confidence : {conf:.1%}")
    print(f"                    Severity   : [{bar}] {sev:.1f}/10  ({sev_lbl(sev)})")

def print_combined(preds, sevs):
    majority = 1 if sum(preds)>len(preds)/2 else 0
    avg      = round(float(np.mean(sevs)),1)
    print(f"\n    {'─'*52}")
    print(f"    COMBINED ASSESSMENT")
    print(f"    {'─'*52}")
    print(f"    Result   : {'⚠️  PTSD DETECTED' if majority else '✅ No PTSD Detected'}")
    print(f"    Severity : {avg:.1f}/10  ({sev_lbl(avg)})")
    print(f"    {'─'*52}")
    if majority:
        print(f"\n    ⚕️  Please consult a mental health professional.\n")


def main():
    text_b   = load_bundle("model/text_bundle.pkl")
    audio_b  = load_bundle("model/audio_bundle.pkl")
    facial_b = load_bundle("model/facial_bundle.pkl")

    if text_b is None:
        print("❌ Models not found. Run: python3 train_model.py")
        sys.exit(1)

    print("=" * 62)
    print("   PTSD MULTIMODAL DETECTION   v4")
    print("=" * 62)
    print(f"   Text model   : {get_name(text_b)}")
    if audio_b:  print(f"   Audio model  : {get_name(audio_b)}")
    if facial_b: print(f"   Facial model : {get_name(facial_b)}")
    print()

    # Demo
    print("─"*62 + "\n  DEMO\n" + "─"*62)
    demos = [
        ("I keep having nightmares about the accident. Every time I "
         "close my eyes I am right back there. Flashbacks are getting "
         "worse and I haven't slept properly in months. I snap at "
         "everyone I love and I can't stop shaking.", True),
        ("Today was a lovely day. I went for a walk in the park, "
         "had coffee with a friend. Feeling grateful and at peace.", False),
    ]
    for raw, sim in demos:
        print(f'\n  Input: "{raw[:70]}..."\n')
        tp,tc,ts = predict_text(text_b, raw)
        print_result("Text", tp, tc, ts)
        preds=[tp]; sevs=[ts]
        if audio_b:
            ap,ac,as_ = predict_structured(audio_b,  sim_audio(sim))
            print_result("Audio*",  ap, ac, as_)
            preds.append(ap); sevs.append(as_)
        if facial_b:
            fp,fc,fs = predict_structured(facial_b, sim_facial(sim))
            print_result("Facial*", fp, fc, fs)
            preds.append(fp); sevs.append(fs)
        print_combined(preds, sevs)

    # Interactive
    print(f"\n{'─'*62}")
    print("  YOUR TURN — type anything to analyse")
    print("  'camera' = live webcam    'quit' = exit")
    print(f"{'─'*62}\n")

    while True:
        try:
            user_input = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!"); break

        if not user_input: continue
        if user_input.lower() in ("quit","exit","q"):
            print("\n  Goodbye!"); break
        if user_input.lower() in ("camera","cam","webcam"):
            os.system("python3 camera_detect.py"); continue

        print()
        tp,tc,ts = predict_text(text_b, user_input)
        print_result("Text", tp, tc, ts)
        preds=[tp]; sevs=[ts]
        if audio_b:
            ap,ac,as_ = predict_structured(audio_b,  sim_audio(tp==1))
            print_result("Audio*",  ap, ac, as_)
            preds.append(ap); sevs.append(as_)
        if facial_b:
            fp,fc,fs = predict_structured(facial_b, sim_facial(tp==1))
            print_result("Facial*", fp, fc, fs)
            preds.append(fp); sevs.append(fs)
        print_combined(preds, sevs)
        print("  (* simulated. Type 'camera' for live facial detection)\n")

if __name__ == "__main__":
    main()