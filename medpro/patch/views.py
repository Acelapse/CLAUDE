"""Portable replacement for Med-Pro's medpro/views.py.

Fixes vs. upstream:
  * Preprocessing now matches training (RGB, pixels scaled to 0-1). Upstream fed raw BGR
    pixels, which roughly halves accuracy on the bundled sample images.
  * argmax over class probabilities instead of exact `== 1.0` comparisons.
  * Works on Windows/macOS/Linux, CPU only (no Windows paths, no numba CUDA reset).
  * Models load once and are cached.
  * Result page shows label, confidence and every class probability, plus a disclaimer.
  * OCR uploads the file the user picked (upstream always sent static/temp/a.jpg) and
    no longer writes to D:\\medpro.
"""
import os, csv, uuid
import cv2, numpy as np, requests, joblib
from django.conf import settings
from django.shortcuts import render

BASE = settings.BASE_DIR
TEMP = os.path.join(BASE, "static", "temp")
PRED = os.path.join(BASE, "static", "pred")

# model file, class keys (training order = alphabetical folder order), readable names
MODELS = {
    "brain":    ("brain_cancer.h5", ["brain_glioma", "brain_menin", "brain_tumor"],
                 ["Glioma", "Meningioma", "Pituitary/other tumor"]),
    "breast":   ("breast_cancer.h5", ["breast_benign", "breast_malignant"], ["Benign", "Malignant"]),
    "all":      ("All_cancer.h5", ["all_benign", "all_early", "all_pre", "all_pro"],
                 ["ALL: benign", "ALL: early", "ALL: pre-B", "ALL: pro-B"]),
    "lymph":    ("lymph_cancer.h5", ["lymph_cll", "lymph_fl", "lymph_mcl"],
                 ["Chronic lymphocytic leukemia", "Follicular lymphoma", "Mantle cell lymphoma"]),
    "kidney":   ("Kidney_cancer.h5", ["kidney_normal", "kidney_tumor"], ["Normal", "Tumor"]),
    "cervical": ("cervical.h5", ["cervix_dyk", "cervix_koc", "cervix_mep", "cervix_pab", "cervix_sfi"],
                 ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-intermediate"]),
    "lung":     ("lung_cancer.h5", ["colon_aca", "colon_bnt", "lung_aca", "lung_bnt", "lung_scc"],
                 ["Colon adenocarcinoma", "Colon benign tissue", "Lung adenocarcinoma",
                  "Lung benign tissue", "Lung squamous cell carcinoma"]),
    "oral":     ("oral_cancer.h5", ["oral_normal", "oral_scc"], ["Normal", "Squamous cell carcinoma"]),
}
_cache = {}

def _model(fname):
    if fname not in _cache:
        import tensorflow as tf
        _cache[fname] = tf.keras.models.load_model(os.path.join(BASE, fname), compile=False)
    return _cache[fname]

def _save_upload(f, sub):
    d = os.path.join(TEMP, sub); os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(f.name)[1].lower() or ".jpg"
    path = os.path.join(d, uuid.uuid4().hex[:10] + ext)
    with open(path, "wb") as out:
        for chunk in f.chunks(): out.write(chunk)
    return path

def image_handle(path):
    img = cv2.imread(path)
    if img is None: raise ValueError("Could not read image")
    img = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(img, 0)

def _predict(request, kind):
    if request.method != "POST" or "img_file" not in request.FILES:
        return render(request, "menu.html")
    fname, keys, names = MODELS[kind]
    path = _save_upload(request.FILES["img_file"], kind)
    try:
        prob = _model(fname).predict(image_handle(path), verbose=0)[0]
    except ValueError:
        return render(request, "disout.html", {"error": "That file is not an image I can read."})
    i = int(np.argmax(prob))
    rows = sorted(zip(names, keys, [float(p) for p in prob]), key=lambda r: -r[2])
    return render(request, "disout.html", {
        "raw": keys[i], "label": names[i], "conf": f"{prob[i]*100:.1f}", "kind": kind,
        "rows": [(n, k, f"{p*100:.1f}") for n, k, p in rows],
        "img": "/static/temp/%s/%s" % (kind, os.path.basename(path)),
    })

def brain_pred(r):    return _predict(r, "brain")
def breast_pred(r):   return _predict(r, "breast")
def all_pred(r):      return _predict(r, "all")
def lymph_pred(r):    return _predict(r, "lymph")
def kidney_pred(r):   return _predict(r, "kidney")
def cervical_pred(r): return _predict(r, "cervical")
def lung_pred(r):     return _predict(r, "lung")
def oral_pred(r):     return _predict(r, "oral")

# ---- simple page views (unchanged) ----
def home(r):       return render(r, "home.html")
def ocrinp(r):     return render(r, "form.html")
def livinp(r):     return render(r, "form2.html")
def threeopt(r):   return render(r, "three.html")
def heart(r):      return render(r, "heart.html")
def cancer(r):     return render(r, "menu.html")
def cancer_pred(r): return render(r, "menu.html")
def brain(r):      return render(r, "form_brain.html")
def breast(r):     return render(r, "form_breast.html")
def all(r):        return render(r, "form_all.html")
def lymph(r):      return render(r, "form_lymph.html")
def kidney(r):     return render(r, "form_kidney.html")
def cervical(r):   return render(r, "form_cervical.html")
def lung(r):       return render(r, "form_lung.html")
def oral(r):       return render(r, "form_oral.html")
def skinpred(r):   return render(r, "form_skin.html")
def segment(r):    return render(r, "form_segmentation.html")
def OCRsubmit(r):  return render(r, "home.html")

# ---- OCR (uses the free ocr.space API, needs internet) ----
def ocr(request):
    lines = []
    if request.method == "POST" and "img_file" in request.FILES:
        path = _save_upload(request.FILES["img_file"], "ocr")
        key = os.environ.get("OCR_SPACE_KEY", "helloworld")  # 'helloworld' = ocr.space public demo key
        try:
            with open(path, "rb") as f:
                r = requests.post("https://api.ocr.space/parse/image", timeout=60,
                                  files={os.path.basename(path): f},
                                  data={"apikey": key, "language": "eng", "isOverlayRequired": False})
            j = r.json()
            text = "" if j.get("IsErroredOnProcessing") else j["ParsedResults"][0]["ParsedText"]
            if not text: lines = ["OCR error: %s" % j.get("ErrorMessage")]
        except Exception as e:
            text, lines = "", ["OCR request failed: %s" % e]
        with open(os.path.join(TEMP, "data.csv"), "w", newline="") as fh:
            csv.writer(fh).writerow([text])
        lines = lines or text.split("\n")
    return render(request, "output2.html", {"raw": lines})

# ---- Liver disease tabular models (sklearn/xgboost pickles from the repo) ----
def liv(request):
    names = ["KNNClassifier_93", "dtr_99", "LRclassifier_71", "rfc_99", "xgb_995"]
    try:
        knn, dtr, lrc, rfc, xgb = [joblib.load(os.path.join(BASE, n)) for n in names]
    except Exception as e:
        return render(request, "output2.html", {"raw": [
            "Liver models could not be loaded with your scikit-learn/xgboost versions:", str(e),
            "Install the versions they were saved with (see README) to use this page."]})
    g = request.POST.get
    sex = g("Gender_of_the_patient", "")
    onehot = [1.0, 0.0, 0.0] if sex == "Male" else [0.0, 1.0, 0.0]
    x = [onehot + [float(g(k)) for k in ("Age_of_the_patient", "Direct_Bilirubin",
         "Alkphos_Alkaline_Phosphotase", "Sgpt_Alamine_Aminotransferase", "Total_Protiens",
         "A_G_Ratio_Albumin_and_Globulin_Ratio")]]
    raw = {k: m.predict(x) for k, m in zip(["knn", "dtr", "lrc", "rfc", "xgb"], [knn, dtr, lrc, rfc, xgb])}
    return render(request, "output1.html" if raw["dtr"] == 1 else "output_.html", {"raw": raw})

# ---- "Segmentation": slides a 128px window and boxes the patch scored most tumor-like ----
def segmentation(request):
    if request.method != "POST" or "img_file" not in request.FILES:
        return render(request, "form_segmentation.html")
    path = _save_upload(request.FILES["img_file"], "brain")
    model = _model("brain_cancer.h5")
    if int(np.argmax(model.predict(image_handle(path), verbose=0)[0])) != 2:
        return render(request, "notumor.html")
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    w, s = 128, 64
    locs = [(i, j) for i in range(0, img.shape[0] - w, s) for j in range(0, img.shape[1] - w, s)]
    if not locs:
        return render(request, "notumor.html")
    batch = np.stack([img[i:i+w, j:j+w] for i, j in locs]).astype("float32") / 255.0
    i, j = locs[int(np.argmax(model.predict(batch, verbose=0)[:, 2]))]
    out = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (j, i), (j + w, i + w), (0, 0, 255), 2)
    os.makedirs(PRED, exist_ok=True)
    name = os.path.basename(path)
    cv2.imwrite(os.path.join(PRED, name), out)
    return render(request, "segimg.html", {"raw": "/static/pred/" + name})
