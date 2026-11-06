#!/usr/bin/env bash
cd "$(dirname "$0")/src" && . ../venv/bin/activate && python manage.py runserver 8000
