#!/bin/bash
set -e
exec gunicorn -c gunicorn.conf.py app:app
