#!/usr/bin/env bash
set -e
exec gunicorn app:app --bind 0.0.0.0:${PORT:-5000}
