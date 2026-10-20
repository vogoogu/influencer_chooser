"""Influencer Chooser: entry point and router. Run with `streamlit run app.py`.

Navigation lives in the header (Workflow and Insights dropdowns) and keeps the original URLs. Each page is a standalone script in pages/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Influencer Chooser", page_icon=str(ROOT / "static" / "icon.svg"), layout="wide",
                   initial_sidebar_state="collapsed")
st.logo(str(ROOT / "static" / "logo.svg"), icon_image=str(ROOT / "static" / "icon.svg"), size="large")

pages = {
    "": [st.Page("pages/0_Home.py", title="Overview", icon=":material/home:", default=True)],
    "Workflow": [
        st.Page("pages/1_Product_Brief.py", title="Product brief", icon=":material/description:", url_path="Product_Brief"),
        st.Page("pages/2_Shortlist.py", title="Shortlist", icon=":material/list_alt:", url_path="Shortlist"),
        st.Page("pages/3_Campaign_Mix.py", title="Campaign mix", icon=":material/tune:", url_path="Campaign_Mix"),
        st.Page("pages/5_Decision_Log.py", title="Decision log", icon=":material/history:", url_path="Decision_Log"),
    ],
    "Insights": [
        st.Page("pages/4_Explorer.py", title="Explorer", icon=":material/search:", url_path="Explorer"),
        st.Page("pages/6_Calibration.py", title="Calibration", icon=":material/target:", url_path="Calibration"),
        st.Page("pages/7_Forecast_Model.py", title="Forecast model", icon=":material/forest:", url_path="Forecast_Model"),
    ],
}
st.navigation(pages, position="top").run()
