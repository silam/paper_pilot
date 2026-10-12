"""
Streamlit UI - UX for AI products.

UX principles applied (good DISCUSSION material):
  * Show SOURCES with every answer  -> users can verify (trust calibration)
  * Show a CONFIDENCE badge          -> groundedness score, honest about uncertainty
  * "Needs review" state             -> transparent Human-in-the-Loop, not a silent failure
  * Thumbs up/down                   -> one-click feedback feeds the improvement loop
  * Debug panel (trace, cost, cache) -> for the demo / for developers
  * Reviewer tab                     -> the human side of HITL

Run:  streamlit run ui/streamlit_app.py   (API must be running on API_URL)
"""
import os

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API}{path}", timeout=120, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        st.error(f"API error: {exc}")
        return None


with st.sidebar:
    st.title("📄 PaperPilot")
    st.caption("Grounded Q&A over AI research papers")
    user_id = st.text_input("User id (drives A/B bucket)", "demo-user")
    version = st.selectbox("Prompt version", ["auto (A/B)", "rag_v1", "rag_v2_fewshot"])
    use_cache = st.toggle("Use cache", True)
    ready = api("GET", "/ready")
    if ready:
        st.success(f"{ready['vectors']} chunks indexed · LLM: {ready['llm_provider']}")
    st.divider()
    st.subheader("Ingest")
    if st.button("Load sample papers"):
        st.json(api("POST", "/ingest", json={"source": "sample"}))
    q = st.text_input("arXiv query", "cat:cs.CL AND abs:RAG")
    if st.button("Fetch from arXiv"):
        st.json(api("POST", "/ingest", json={"source": "arxiv", "query": q, "max_results": 15}))
    up = st.file_uploader("Upload PDF / MD / TXT", type=["pdf", "md", "txt"])
    if up and st.button("Index upload"):
        st.json(api("POST", "/upload", files={"file": (up.name, up.getvalue())}))

tab_ask, tab_review, tab_stats = st.tabs(["Ask", "Human review", "A/B stats"])

with tab_ask:
    question = st.text_input("Ask about the papers", "Why does LoRA add no inference latency?")
    if st.button("Ask", type="primary") and question:
        payload = {"question": question, "user_id": user_id, "use_cache": use_cache}
        if version != "auto (A/B)":
            payload["prompt_version"] = version
        st.session_state["last"] = api("POST", "/ask", json=payload)

    res = st.session_state.get("last")
    if res:
        g = res["groundedness"]
        badge = "🟢 High" if g >= 0.8 else "🟡 Medium" if g >= 0.45 else "🔴 Low"
        if res["status"] == "blocked":
            st.error(f"Blocked by guardrails: {', '.join(res['guard_reasons'])}")
        elif res["status"] == "needs_review":
            st.warning("Low confidence - this answer was sent to a human reviewer.")
        st.markdown(f"### Answer  \n{res['answer']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confidence", badge, f"{g:.2f}")
        c2.metric("Latency", f"{res['latency_ms']:.0f} ms")
        c3.metric("Cost", f"${res['cost_usd']:.5f}")
        c4.metric("Cache", res["cache"])
        st.caption(f"prompt={res['prompt_version']} · model={res['model']} · trace={res['trace_id']}")

        for s in res["sources"]:
            with st.expander(f"[{s['n']}] {s['title']}  (score {s['score']})"):
                st.write(s["snippet"] + "…")
                if s["url"]:
                    st.markdown(f"[Open paper]({s['url']})")

        fb1, fb2, _ = st.columns([1, 1, 6])
        if fb1.button("👍"):
            api("POST", "/feedback", json={"query_id": res["query_id"], "rating": 1})
            st.toast("Thanks!")
        if fb2.button("👎"):
            api("POST", "/feedback", json={"query_id": res["query_id"], "rating": -1})
            st.toast("Sent to review queue")
        with st.expander("🔍 Trace (per-step latency)"):
            st.dataframe(res["trace"], use_container_width=True)

with tab_review:
    st.subheader("Pending human review")
    for item in api("GET", "/review") or []:
        with st.container(border=True):
            st.markdown(f"**Q:** {item['question']}  \n**Model answer:** {item['answer']}")
            st.caption(f"reason: {item['reason']} · groundedness {item['groundedness']}")
            fix = st.text_area("Corrected answer", key=f"fix{item['review_id']}")
            a, b = st.columns(2)
            if a.button("Approve", key=f"ap{item['review_id']}"):
                api("POST", f"/review/{item['review_id']}", json={"action": "approve"})
                st.rerun()
            if b.button("Save correction", key=f"co{item['review_id']}"):
                api("POST", f"/review/{item['review_id']}", json={"action": "correct", "answer": fix})
                st.rerun()

with tab_stats:
    st.subheader("Prompt A/B results (live traffic)")
    st.dataframe(api("GET", "/stats/ab") or [], use_container_width=True)
