"""Run each Med-Pro .h5 model on the repo's labeled sample images.
Usage: python test_models.py <path-to-Med_pro_webApp> [extra_image_dir]
Filenames must start with the class label, e.g. brain_glioma_0001.jpg."""
import os, sys, glob, cv2, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

# Class order copied from medpro/views.py
MODELS = {
  "brain_cancer.h5":  ["brain_glioma", "brain_menin", "brain_tumor"],
  "breast_cancer.h5": ["breast_benign", "breast_malignant"],
  "All_cancer.h5":    ["all_benign", "all_early", "all_pre", "all_pro"],
  "lymph_cancer.h5":  ["lymph_cll", "lymph_fl", "lymph_mcl"],
  "Kidney_cancer.h5": ["kidney_normal", "kidney_tumor"],
  "cervical.h5":      ["cervix_dyk", "cervix_koc", "cervix_mep", "cervix_pab", "cervix_sfi"],
  "lung_cancer.h5":   ["colon_aca", "colon_bnt", "lung_aca", "lung_bnt", "lung_scc"],
  "oral_cancer.h5":   ["oral_normal", "oral_scc"],
}

def load(path):  # matches training: RGB, 128x128, pixels scaled to 0-1
    img = cv2.cvtColor(cv2.resize(cv2.imread(path), (128, 128)), cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype("float32") / 255.0, 0)


root = sys.argv[1]
imgs = sorted(glob.glob(os.path.join(root, "static/temp/**/*.jpg"), recursive=True))
if len(sys.argv) > 2:
    imgs += sorted(glob.glob(os.path.join(sys.argv[2], "*.jpg")))
for mfile, classes in MODELS.items():
    model = tf.keras.models.load_model(os.path.join(root, mfile), compile=False)
    mine = {}
    for p in imgs:
        name = os.path.basename(p)
        label = next((c for c in classes if name.startswith(c)), None)
        if label: mine[name] = (p, label)
    print(f"\n== {mfile}  ({len(mine)} labeled samples)")
    ok = 0
    for name, (p, label) in sorted(mine.items()):
        prob = model.predict(load(p), verbose=0)[0]
        pred = classes[int(np.argmax(prob))]
        ok += pred == label
        print(f"  {'OK ' if pred == label else 'BAD'} {name:28s} -> {pred:16s} conf={prob.max():.3f}")
    if mine: print(f"  accuracy {ok}/{len(mine)}")
    # sanity check: a non-medical random image should not be confidently classified
    noise = np.random.randint(0, 255, (1, 128, 128, 3)).astype("float32") / 255.0
    pn = model.predict(noise, verbose=0)[0]
    print(f"  random-noise image -> {classes[int(np.argmax(pn))]} conf={pn.max():.3f}")
