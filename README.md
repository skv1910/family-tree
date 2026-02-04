# Family Tree (Streamlit)

Create and extend a family tree with photos and names. Add parents or children from the UI, and store images directly in the repo for easy hosting on Streamlit Community Cloud.

## Features
- Top-down family tree visualization
- Add parents or children from the sidebar
- Optional photo upload per person
- Placeholder image shown when no photo is provided

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally
```bash
streamlit run app.py
```

## Data and images
- Family data is stored in `data/family.json`
- Uploaded images are saved in `assets/images/`
- A placeholder image is created automatically at `assets/placeholder.png` if it does not exist

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app and select this repo.
3. Set the main file path to `app.py`.
4. Deploy.

The `packages.txt` file ensures the Graphviz system dependency is installed on Streamlit Cloud.
