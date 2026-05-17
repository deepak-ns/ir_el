# ui/app.py
"""
Streamlit frontend for full-PDF search.
"""

import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import cfg


API_BASE = cfg.ui.api_base_url


st.set_page_config(
    page_title="PDF Search",
    page_icon="PDF",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.result-card { border:1px solid #e0e0e0; border-radius:8px; padding:1rem 1.25rem; margin-bottom:1rem; background:#ffffff; }
.rank-badge  { font-size:0.75rem; font-weight:600; color:#6c5ce7; background:#eeedfe; padding:2px 8px; border-radius:12px; }
.score-bar-text { font-size:0.75rem; color:#636e72; }
.query-type-badge { font-size:0.8rem; font-weight:500; padding:3px 10px; border-radius:12px; }
.qt-text        { background:#e1f5ee; color:#0f6e56; }
.qt-cross_modal { background:#eeedfe; color:#3c3489; }
.qt-figure      { background:#faece7; color:#712b13; }
</style>
""", unsafe_allow_html=True)


def _load_image(url: str):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        return None
    return None


def _render_results(data: dict):
    results = data.get("results", [])
    qt = data.get("query_type", "text")
    explanation = data.get("routing_explanation", "")
    latency = data.get("latency_ms", 0)
    n_cands = data.get("total_candidates", 0)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        qt_label = {"text": "Text search", "cross_modal": "Visual text", "figure": "Figure search"}.get(qt, qt)
        st.markdown(f'<span class="query-type-badge qt-{qt}">{qt_label}</span>', unsafe_allow_html=True)
        st.caption(explanation)
    with col2:
        st.metric("Results", len(results))
    with col3:
        st.metric("Latency", f"{latency} ms")

    st.caption(f"Candidates: {n_cands} | alpha={data.get('alpha', 0):.2f}, beta={data.get('beta', 0):.2f}")
    st.divider()

    if not results:
        st.info("No results found. Try a different keyword or phrase.")
        return

    for r in results:
        doc = r["paper"]
        matched_page = r.get("matched_page")
        matched_figs = r.get("matched_figures", [])
        breakdown = r["score_breakdown"]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col_rank, col_title = st.columns([0.06, 0.94])
        with col_rank:
            st.markdown(f'<span class="rank-badge">#{r["rank"]}</span>', unsafe_allow_html=True)
        with col_title:
            st.markdown(f"**{doc['title']}**")
            meta = []
            if doc.get("authors"):
                meta.append(", ".join(doc["authors"][:3]) + (" et al." if len(doc["authors"]) > 3 else ""))
            if doc.get("year"):
                meta.append(str(doc["year"]))
            if doc.get("venue"):
                meta.append(doc["venue"])
            if meta:
                st.caption(" · ".join(meta))

        left, right = st.columns([0.58, 0.42]) if matched_page or matched_figs else (st.container(), None)

        with left:
            if matched_page:
                st.caption(f"Best matching page: {matched_page.get('page_number')}")
                page_text = matched_page.get("text", "")
                st.markdown(page_text[:900] + ("..." if len(page_text) > 900 else ""))
            else:
                sample = doc.get("abstract", "") or ""
                st.markdown(sample[:500] + ("..." if len(sample) > 500 else ""))

        if right and matched_page:
            with right:
                image_url = matched_page.get("image_url")
                if image_url:
                    img = _load_image(f"{API_BASE}{image_url}")
                    if img:
                        st.image(img, caption=f"Page {matched_page.get('page_number')}", use_container_width=True)

        elif right and matched_figs:
            with right:
                st.caption("Matched figures")
                for fig in matched_figs[:2]:
                    image_url = fig.get("image_url")
                    if image_url:
                        img = _load_image(f"{API_BASE}{image_url}")
                        if img:
                            st.image(img, caption=fig.get("caption", "")[:80], use_container_width=True)
                    else:
                        st.caption(fig.get("caption", "")[:80])

        st.markdown('<div class="score-bar-text">Score breakdown</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.progress(breakdown["text_pct"] / 100, text=f"Text {breakdown['text_pct']}%")
        with s2:
            st.progress(breakdown["figure_pct"] / 100, text=f"Figure {breakdown['figure_pct']}%")
        with s3:
            st.progress(breakdown["bm25_pct"] / 100, text=f"BM25 {breakdown['bm25_pct']}%")

        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()


with st.sidebar:
    st.title("Search Settings")
    st.markdown("#### Fusion weights")
    alpha = st.slider("Text weight", 0.0, 1.0, 0.75, 0.05)
    beta = st.slider("Figure weight", 0.0, 1.0, 0.25, 0.05)
    top_k = st.slider("Results to show", 3, 20, 10)
    st.divider()
    st.markdown("#### Index stats")
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=3).json()
        st.metric("PDFs indexed", stats.get("papers_in_db", "-"))
        st.metric("Pages indexed", stats.get("pages_in_db", "-"))
        st.metric("Figures indexed", stats.get("figures_in_db", "-"))
    except Exception:
        st.warning("API offline - run `python scripts/run.py`")


st.title("PDF Search")
st.caption("Upload PDFs, search their full text, and view the best matching page.")

tab_text, tab_figure, tab_upload = st.tabs([
    "Text search",
    "Figure image upload",
    "Upload PDF",
])


with tab_text:
    st.markdown("Search for any keyword or phrase from the uploaded PDFs.")
    query = st.text_input(
        "Query",
        placeholder="invoice number, policy clause, model architecture, payment terms",
        label_visibility="collapsed",
    )
    search_btn = st.button("Search", type="primary", key="text_search")

    if search_btn and query:
        with st.spinner("Searching..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/search",
                    json={"query": query, "alpha": alpha, "beta": beta, "top_k": top_k},
                    timeout=30,
                )
                resp.raise_for_status()
                _render_results(resp.json())
            except requests.ConnectionError:
                st.error("Cannot connect to API. Run `python scripts/run.py` first.")
            except Exception as e:
                st.error(f"Search error: {e}")


with tab_figure:
    st.markdown("Upload an image to find PDFs with visually similar extracted figures.")
    uploaded_fig = st.file_uploader("Figure image", type=["png", "jpg", "jpeg"])
    fig_caption = st.text_input("Optional caption / description", key="fig_cap")
    fig_btn = st.button("Search by figure", type="primary", key="fig_search")

    if fig_btn and uploaded_fig:
        with st.spinner("Encoding with CLIP..."):
            try:
                files = {"file": (uploaded_fig.name, uploaded_fig.getvalue(), uploaded_fig.type)}
                params = {"top_k": top_k, "alpha": alpha, "beta": beta}
                if fig_caption:
                    params["caption"] = fig_caption
                resp = requests.post(f"{API_BASE}/search/figure", files=files, params=params, timeout=60)
                resp.raise_for_status()
                _render_results(resp.json())
            except requests.ConnectionError:
                st.error("Cannot connect to API.")
            except Exception as e:
                st.error(f"Search error: {e}")


with tab_upload:
    st.markdown("### Add a PDF to the index")
    st.markdown(
        "Upload any text-based PDF. Every page will be parsed, embedded, and added to the searchable index."
    )

    col_form, col_info = st.columns([0.55, 0.45])

    with col_form:
        uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_upload")

        with st.expander("Optional metadata", expanded=False):
            user_title = st.text_input("Title", placeholder="Auto-extracted from PDF if blank")
            user_authors = st.text_input("Authors", placeholder="Alice Smith, Bob Jones")
            user_year = st.number_input("Year", min_value=1900, max_value=2100, step=1)
            user_year = int(user_year) if user_year else None

        add_btn = st.button("Add to index", type="primary", key="add_pdf")

    with col_info:
        st.markdown("**What happens when you click Add:**")
        st.markdown("""
1. Text is extracted from every page
2. Each page becomes a Sentence-BERT vector
3. BM25 keyword search is rebuilt over all pages
4. Page images are rendered for result previews
5. The PDF is searchable immediately
        """)
        st.info("Takes about 10-60 seconds on CPU depending on length.")

    if add_btn and uploaded_pdf:
        progress = st.progress(0, text="Saving PDF...")

        try:
            progress.progress(10, text="Uploading PDF to API...")
            files = {
                "file": (uploaded_pdf.name, uploaded_pdf.getvalue(), uploaded_pdf.type or "application/pdf")
            }
            params = {}
            if user_title:
                params["title"] = user_title
            if user_authors:
                params["authors"] = user_authors
            if user_year:
                params["year"] = user_year

            progress.progress(35, text="Parsing, embedding, and updating live index...")
            resp = requests.post(f"{API_BASE}/ingest", files=files, params=params, timeout=180)
            resp.raise_for_status()
            data = resp.json()

            progress.progress(100, text="Done!")
            st.success(f"**{uploaded_pdf.name}** has been added to the live index.")
            st.info("Switch to the Text search tab and search for any keyword or phrase from the PDF.")

            with st.expander("What was extracted from your PDF", expanded=True):
                st.markdown(f"**Document ID:** {data['paper_id']}")
                st.markdown(f"**Title:** {data['title']}")
                st.markdown(f"**Text sample:** {data['abstract']}...")
                st.markdown(f"**Pages indexed:** {data.get('pages_indexed', 0)}")
                st.markdown(f"**Figures found:** {data['figures_found']}")

        except requests.ConnectionError:
            progress.progress(100, text="Failed")
            st.error("Cannot connect to API. Run `python scripts/run.py` first.")
        except requests.HTTPError as e:
            progress.progress(100, text="Failed")
            detail = e.response.text if e.response is not None else str(e)
            st.error(f"Indexing failed: {detail}")
        except Exception as e:
            progress.progress(100, text="Failed")
            st.error(f"Error: {e}")

    elif add_btn and not uploaded_pdf:
        st.warning("Please select a PDF file first.")

    st.divider()
    st.markdown("#### Already indexed PDFs")
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=3).json()
        n = stats.get("papers_in_db", 0)
        pages = stats.get("pages_in_db", 0)
        st.caption(f"{n} PDF(s), {pages} page(s) currently in the index.")
    except Exception:
        st.caption("Start the API to see index stats.")
