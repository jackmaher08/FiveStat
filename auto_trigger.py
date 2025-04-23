import sys
import os

def main():
    print("🧪 Script has started running")
    print("🔍 Checking for new data...")

    existing_df = load_existing_data()
    understat_df = fetch_understat_fixtures()

    if new_data_available(existing_df, understat_df):
        print("✅ New data detected — running update pipeline.")

        # Run Python steps directly (no .sh needed)
        os.system(f"{sys.executable} data_scraper_script.py")
        os.system(f"{sys.executable} data_loader.py")
        os.system(f"{sys.executable} generate_radars.py")
        os.system(f"{sys.executable} generate_shotmaps.py")

        # Git commit and push
        os.system("git add .")
        os.system
