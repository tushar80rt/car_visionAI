import os
import json
from dotenv import load_dotenv
from labellerr.client import LabellerrClient
from labellerr.exceptions import LabellerrError

# Load API credentials
load_dotenv("api.env")

LABELLERR_CLIENT_ID = os.getenv("LABELLERR_CLIENT_ID")
LABELLERR_EMAIL = os.getenv("LABELLERR_EMAIL")
LABELLERR_API_KEY = os.getenv("LABELLERR_API_KEY")
LABELLERR_API_SECRET = os.getenv("LABELLERR_API_SECRET")

DATA_TYPE = "image"
PROJECT_NAME = "detect car v4"
DATASET_NAME = "detect car v4"
DATASET_DESCRIPTION = "10 sample vehicle images"
FOLDER_TO_UPLOAD = "data/samples"

ANNOTATION_QUESTIONS = [
    {
        "question_number": 1,
        "question": "Car",
        "question_id": "car-bbox-001",
        "option_type": "BoundingBox",
        "required": False,
        "options": [{"option_id": "opt-001", "option_name": "#FF0000"}],
        "question_metadata": [],
    }
]

try:
    print("🚀 Initializing Labellerr Client...")
    client = LabellerrClient(LABELLERR_API_KEY, LABELLERR_API_SECRET)
    print("✅ Client initialized successfully!")

    print("📋 Creating annotation guideline...")
    template_id = client.create_annotation_guideline(
        client_id=LABELLERR_CLIENT_ID,
        questions=ANNOTATION_QUESTIONS,
        template_name=f"{PROJECT_NAME} Template",
        data_type=DATA_TYPE,
    )
    print(f"✅ Template created: {template_id}")

    # --- Step 1: Dataset create karna ---
    payload = {
        "client_id": LABELLERR_CLIENT_ID,
        "dataset_name": DATASET_NAME,
        "dataset_description": DATASET_DESCRIPTION,
        "data_type": DATA_TYPE,
        "created_by": LABELLERR_EMAIL,
        "project_name": PROJECT_NAME,
        "annotation_template_id": template_id,
        "rotation_config": {
            "annotation_rotation_count": 1,
            "review_rotation_count": 1,
            "client_review_rotation_count": 1,
        },
        "autolabel": False,
        "folder_to_upload": FOLDER_TO_UPLOAD,
    }

    print("🚀 Creating dataset via initiate_create_project...")
    res = client.initiate_create_project(payload)
    print("✅ Dataset created successfully (check your Labellerr dashboard).")
    print("👉 Go to your dashboard to annotate:")
    print("   https://app.labellerr.com/tool/dashboard")

    # --- Step 2: Link existing dataset manually ---
    dataset_id = input("\n🔗 Enter your existing dataset_id to link with new project: ").strip()

    if dataset_id:
        rotation_config = {
            "annotation_rotation_count": 1,
            "review_rotation_count": 1,
            "client_review_rotation_count": 1,
        }

        print("\n🚀 Creating project and linking existing dataset...")
        res2 = client.create_project(
            project_name=PROJECT_NAME,
            data_type=DATA_TYPE,
            client_id=LABELLERR_CLIENT_ID,
            dataset_id=dataset_id,
            annotation_template_id=template_id,
            rotation_config=rotation_config,
        )

        print("✅ Project created successfully and linked with dataset!")
        print(json.dumps(res2, indent=2))
    else:
        print("⚠️ No dataset_id entered. Skipping project linking step.")

except LabellerrError as e:
    print(f"❌ Labellerr Error: {e}")
except Exception as e:
    print(f"⚠️ Unexpected Error: {e}")
