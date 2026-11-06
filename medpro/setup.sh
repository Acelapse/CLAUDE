#!/usr/bin/env bash
# Mac/Linux: downloads Med-Pro, applies the fixes in patch/, installs everything.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d src ] || git clone --depth 1 https://github.com/kumar11jr/Med_pro_webApp.git src
cp patch/views.py src/medpro/views.py
cp patch/templates/disout.html src/templates/disout.html
[ -d venv ] || $PY -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
(cd src && python manage.py migrate -v0)
echo; echo "Done. Start it with:  ./run.sh   then open http://localhost:8000"
