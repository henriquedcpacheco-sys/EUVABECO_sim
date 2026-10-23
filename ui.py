"""Page furniture shared by every mode: header, sidebar footer, empty states,
chart grids and a small amount of CSS."""
from pathlib import Path

import pandas as pd
import streamlit as st

from data import NPI_EVENTS

DATA_PATH = Path(__file__).parent / "data"
MODEL_REF_PDF = DATA_PATH / "model_reference.pdf"

# Fixed NPI changepoints: (date, event), drawn as dashed rules in every chart.
CHANGEPOINTS = [(pd.Timestamp(e[2]), e[0]) for e in NPI_EVENTS]

_CSS = """
<style>
/* Tighter top of the page; the header carries its own spacing. */
[data-testid="stMainBlockContainer"] { padding-top: 4.4rem !important; max-width: 1320px; }
.app-eyebrow {
  text-transform: uppercase; letter-spacing: .09em; font-size: .74rem;
  font-weight: 600; opacity: .62; margin-bottom: .1rem;
}
.app-lede { font-size: 1.04rem; opacity: .86; max-width: 60rem; margin: .1rem 0 .35rem; }
.app-authors { font-size: .9rem; opacity: .72; }
/* Section tabs: a little more presence than the default. */
div[data-testid="stTabs"] > div[role="tablist"] button p { font-size: .98rem; font-weight: 500; }
/* Sidebar: lighter radio captions, small headings. */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { opacity: .78; }
section[data-testid="stSidebar"] h4 { margin-bottom: .2rem; }
.sb-footer { font-size: .8rem; opacity: .72; line-height: 1.5; }
.sb-footer a { text-decoration: none; }
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


@st.dialog("Model scheme and equations", width="large")
def _show_model_reference():
    # Pre-rendered PNG pages: browsers block data:-URL PDFs in tabs and iframes.
    for page in sorted(DATA_PATH.glob("model_reference-*.png")):
        st.image(str(page))


def render_header():
    st.markdown('<div class="app-eyebrow">EUVABECO · CEMAT · Online COVID-19 simulator</div>',
                unsafe_allow_html=True)
    st.title("COVID-19 Model Simulator — Portugal")
    st.markdown(
        '<div class="app-lede">Explore how a compartmental model fitted to Portugal\'s '
        "2020 epidemic reproduces confirmed cases, ward and ICU occupancy and deaths, "
        "compare model versions of increasing complexity, and test counterfactual "
        "second-wave lockdowns.</div>"
        '<div class="app-authors"><b>Henrique Pacheco</b>, CEMAT &nbsp;·&nbsp; '
        "<b>Erida Gjini</b>, CEMAT*</div>",
        unsafe_allow_html=True,
    )
    row = st.container(horizontal=True, gap="small", vertical_alignment="center")
    with row:
        with st.popover("About this simulator", icon=":material/info:"):
            st.markdown(ABOUT_TEXT)
        if MODEL_REF_PDF.exists():
            if st.button("Model scheme and equations", icon=":material/schema:"):
                _show_model_reference()
            st.download_button(
                "Equations (PDF)", data=MODEL_REF_PDF.read_bytes(),
                file_name="model_reference.pdf", mime="application/pdf",
                icon=":material/download:", on_click="ignore",
            )
    st.write("")


ABOUT_TEXT = """
#### Welcome to the Online COVID-19 Simulator

We provide a tool in which it is possible to explore how COVID-19 spreads in
a particular setting, and how different factors influence the course of an
outbreak, through our interactive online simulator. We have adjusted the
model to epidemiological data from Portugal
([DSSG Portugal](https://github.com/dssg-pt/covid19pt-data);
[Our World in Data](https://github.com/owid/covid-19-data)), especially the
early phase of the SARS-CoV-2 pandemic. We allow several model versions,
based on [Dings et al. (2022)](https://doi.org/10.3390/v14102114), of
increasing complexity to be employed, where several parameters such as
transmission rates, hospitalisation probabilities and others are estimated
to adjust model predictions to the observed time series.

In addition to model calibration based on early dynamics, it is possible to
investigate hypothetical counterfactual scenarios and observe the impact of
different parameters on infection patterns over time. We also implement
models with vaccination for later stages of the pandemic. Although we only
model aggregated nation-level data for Portugal, the platform can be flexibly
extended to model specific regional data of the same type and resolution.
"""


def sidebar_footer(sb):
    sb.divider()
    sb.markdown(
        '<div class="sb-footer">'
        "Data: <a href='https://github.com/dssg-pt/covid19pt-data'>DSSG Portugal</a>, "
        "<a href='https://github.com/owid/covid-19-data'>Our World in Data</a><br>"
        "Model based on <a href='https://doi.org/10.3390/v14102114'>Dings et al. (2022)</a><br>"
        "Part of the EUVABECO project"
        "</div>",
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, text: str):
    """A quiet placeholder card. `icon` is a ':material/name:' string."""
    with st.container(border=True):
        st.markdown(f"##### {icon} {title}")
        st.markdown(text)


def panel_grid(panels: list, ncols: int = 2):
    """Draw chart panels in a responsive grid (columns stack on narrow screens)."""
    for i in range(0, len(panels), ncols):
        cols = st.columns(ncols, gap="large")
        for col, p in zip(cols, panels[i:i + ncols]):
            col.altair_chart(p, width="stretch")
