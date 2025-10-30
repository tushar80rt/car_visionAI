# --- Car Dataset Export Script ---
import os
import json
import time
import uuid
import logging
import traceback
import requests
from pathlib import Path
from dotenv import load_dotenv
from labellerr.client import LabellerrClient
from labellerr.exceptions import LabellerrError

# --- Load Credentials from api.env ---
load_dotenv("api.env")

LABELLERR_CLIENT_ID = os.getenv("LABELLERR_CLIENT_ID")
LABELLERR_API_KEY = os.getenv("LABELLERR_API_KEY")
LABELLERR_API_SECRET = os.getenv("LABELLERR_API_SECRET")

# --- Ask for Project ID ---
project_id = input("Enter your Labellerr Project ID: ").strip()

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger(__name__)

# --- INIT CLIENT ---
client = LabellerrClient(LABELLERR_API_KEY, LABELLERR_API_SECRET)

# ✅ Step 1: Create Export
def create_export():
    export_config = {
        "export_name": "CarDatasetExport",
        "export_description": "Export of all annotated car images",
        "export_format": "json",
        "statuses": ["review", "r_assigned", "client_review", "cr_assigned", "accepted"],
        "export_destination": "local",
        "question_ids": ["all"],
    }

    try:
        logger.info("🚀 Creating export job...")
        res = client.create_local_export(
            project_id=project_id,
            client_id=LABELLERR_CLIENT_ID,
            export_config=export_config
        )
        export_id = res["response"]["report_id"]
        logger.info(f"✅ Export created successfully! Export ID: {export_id}")
        return export_id
    except LabellerrError as e:
        logger.error(f"❌ Export creation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error during export creation: {e}")
        return None


# ✅ Step 2: Poll Export Status
def poll_export_status(export_id, max_wait_time=300, wait_interval=10):
    logger.info("⏳ Waiting for export to complete...")
    elapsed_time = 0

    while elapsed_time < max_wait_time:
        try:
            status_res = client.check_export_status(
                api_key=LABELLERR_API_KEY,
                api_secret=LABELLERR_API_SECRET,
                project_id=project_id,
                report_ids=[export_id],
                client_id=LABELLERR_CLIENT_ID
            )

            if isinstance(status_res, str):
                status_res = json.loads(status_res)

            status = status_res["status"][0]
            is_done = status.get("is_completed", False)
            state = status.get("export_status", "unknown")

            logger.info(f"📦 Export status: {state}")
            if is_done:
                logger.info("✅ Export completed successfully!")
                return True
            elif state.lower() == "failed":
                logger.error("❌ Export failed!")
                return False

        except Exception as e:
            logger.warning(f"⚠️ Error checking status: {e}")

        time.sleep(wait_interval)
        elapsed_time += wait_interval

    logger.warning("⌛ Timeout waiting for export completion.")
    return False


# ✅ Step 3: Download Export File
def download_export(export_id):
    try:
        download_uuid = str(uuid.uuid4())
        result = client.fetch_download_url(
            api_key=LABELLERR_API_KEY,
            api_secret=LABELLERR_API_SECRET,
            project_id=project_id,
            uuid=download_uuid,
            export_id=export_id,
            client_id=LABELLERR_CLIENT_ID
        )

        if isinstance(result, str):
            result = json.loads(result)

        download_url = result.get("url") or result.get("response", {}).get("download_url")

        if not download_url:
            logger.error(f"❌ No download URL found: {result}")
            return None

        logger.info(f"🔗 Download URL fetched: {download_url}")

        exports_dir = Path("exports")
        exports_dir.mkdir(exist_ok=True)
        export_path = exports_dir / f"car_dataset_export_{export_id}.json"

        response = requests.get(download_url)
        if response.status_code == 200:
            with open(export_path, "wb") as f:
                f.write(response.content)
            logger.info(f"💾 Export file saved to: {export_path}")
            return export_path
        else:
            logger.error(f"❌ Download failed, HTTP {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"❌ Error downloading export: {e}\n{traceback.format_exc()}")
        return None


# ✅ Step 4: Validate JSON
def validate_json(export_path):
    try:
        with open(export_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total = len(data)
        annotated = sum(1 for x in data if x.get("latest_answer"))
        logger.info(f"📊 Annotated: {annotated}/{total}")
        return True
    except Exception as e:
        logger.error(f"❌ JSON validation error: {e}")
        return False


# 🚀 MAIN FLOW
if __name__ == "__main__":
    logger.info("=== Car Dataset Export Script Started ===")
    export_id = create_export()

    if export_id:
        if poll_export_status(export_id):
            export_file = download_export(export_id)
            if export_file:
                validate_json(export_file)
    else:
        logger.error("❌ Export creation failed, stopping script.")
