@echo off
cd /d C:\Users\jmaher\Documents\flask_heatmap_app
python data_loader.py
git add .
git commit -m "Automated data update"
git push origin main
exit