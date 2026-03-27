import streamlit as st
from graph import app, llm
from langchain_core.prompts import ChatPromptTemplate
import os

# Page Config
st.set_page_config(page_title="Agentic Research Assistant", layout="centered")

# App Header
st.title("📄Scholar Research Assistant")
st.write("Automatically research papers from ArXiv and generate a summarized report.")

# Session State for Chat History and Agent State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Agentic AI Research Assistant. What field are we researching today? (e.g., Physics, Computer Science, Renewable Energy)"}]
if "research_state" not in st.session_state:
    st.session_state.research_state = {
        "topic": None,
        "papers": [],
        "selected_indices": None,
        "selected_papers": [],
        "paper_summaries": {},
        "ideas": [],
        "final_paper_path": None,
        "final_tex_path": None,
        "pdf_data": None,
        "tex_data": None
    }

# Helper to get short abstract + keywords from the LLM
def get_short_abstract_and_keywords(abstract: str, paper_index: int) -> dict:
    cache = st.session_state.research_state.get("paper_summaries", {})
    if paper_index in cache:
        return cache[paper_index]

    prompt_template = """
Given the paper abstract, provide these fields:
- Short Abstract (2-3 sentences)
- Main Focus (e.g., NLP, computer vision, graph neural networks, face recognition, etc.)
- Keywords

Output format:
Short Abstract: ...
Main Focus: ...
Keywords: kw1, kw2, kw3, ...

Abstract:
{abstract}
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    try:
        response = chain.invoke({"abstract": abstract})
        content = response.content.strip()
    except Exception as e:
        content = f"( summary unavailable: {e})"

    summary = {
        "short_abstract": "",
        "main_focus": "",
        "keywords": "",
        "raw": content
    }

    for line in content.splitlines():
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip().lower()
        value = parts[1].strip()
        if key == "short abstract":
            summary["short_abstract"] = value
        elif key == "main focus":
            summary["main_focus"] = value
        elif key == "keywords":
            summary["keywords"] = value

    # if model gave minimal plain text, fallback the raw content
    if not summary["short_abstract"]:
        summary["short_abstract"] = content

    cache[paper_index] = summary
    st.session_state.research_state["paper_summaries"] = cache
    return summary


def compute_match_percentage(term: str, summary: dict, paper: dict) -> int:
    if not term:
        return 0

    term = term.strip().lower()
    focus = summary.get("main_focus", "").lower()
    keywords = summary.get("keywords", "").lower()
    title = paper.get("title", "").lower()
    abstract = paper.get("summary", "").lower()

    score = 0

    if term in focus:
        score += 40
    if term in keywords:
        score += 30
    if term in title:
        score += 20
    if term in abstract:
        score += 10

    # If no direct full-term match but partial token matches exist, give partial score.
    term_tokens = term.split()
    token_matches = 0
    for tok in term_tokens:
        if tok and (tok in focus or tok in keywords or tok in title or tok in abstract):
            token_matches += 1
    if token_matches and score == 0:
        score = min(100, 20 + 20 * token_matches)

    return min(100, score)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input via Chat
if prompt := st.chat_input("Type your research topic or follow-up..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. If topic is not decided yet
        if not st.session_state.research_state["topic"]:
            st.session_state.research_state["topic"] = prompt
            with st.spinner(f"Searching for recent papers on {prompt}..."):
                initial_state = {
                    "research_topic": prompt,
                    "paper_results": [],
                    "selected_papers": [],
                    "analyses": [],
                    "final_manuscript": "",
                    "final_report_path": "",
                    "latex_report_path": "",
                    "tectonic_pdf_path": None,
                    "status": "starting"
                }
                # Use the search node from our graph
                from graph import search_papers_node
                result = search_papers_node(initial_state)
                st.session_state.research_state["papers"] = result["paper_results"]
                
                response = f"I found these 10 recent papers on **{prompt}**. Expand each to see an AI-generated short abstract, main focus, and keywords. Then choose which ones you'd like me to analyze (e.g., enter '1, 3, 5, 6, 7, 8' to pick 6, or 'all' for all 10).\n\nTip: To narrow to your interest area, type 'filter <topic>' (e.g., 'filter nlp' or 'filter face recognition')."
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display papers with expanders for previews
                search_term = st.session_state.research_state.get("topic", "")
                for i, p in enumerate(result["paper_results"]):
                    with st.expander(f"{i+1}. {p['title']} ({p['published']})"):
                        short_summary = get_short_abstract_and_keywords(p['summary'], i)
                        match_score = compute_match_percentage(search_term, short_summary, p)
                        st.markdown(f"**Match Percentage (vs. your topic '{search_term}')**: {match_score}%")
                        st.markdown("**AI Summary:**")
                        st.write(short_summary.get("short_abstract", ""))
                        st.markdown("**Main Focus:**")
                        st.write(short_summary.get("main_focus", "(not determined)"))
                        st.markdown("**Keywords:**")
                        st.write(short_summary.get("keywords", "(none)"))
                        st.markdown("---")
                        with st.expander("Show full original abstract"):
                            st.write(p['summary'])
        # 2. If topic is decided but papers are not selected
        elif not st.session_state.research_state["selected_indices"]:
            try:
                import re
                user_input = prompt.strip()
                user_input_lc = user_input.lower()

                # Filter mode based on focus or keywords
                if user_input_lc.startswith("filter"):
                    filter_term = user_input.split(None, 1)[1].strip() if len(user_input.split(None, 1)) > 1 else ""
                    if not filter_term:
                        st.error("Please provide a term to filter by, e.g., 'filter nlp'.")
                    else:
                        st.markdown(f"### 📌 Filtering papers for: '{filter_term}'")
                        matches = 0
                        for i, p in enumerate(st.session_state.research_state["papers"]):
                            summary = get_short_abstract_and_keywords(p['summary'], i)
                            focus = summary.get("main_focus", "").lower()
                            keywords = summary.get("keywords", "").lower()
                            if filter_term.lower() in focus or filter_term.lower() in keywords or filter_term.lower() in p['title'].lower() or filter_term.lower() in p['summary'].lower():
                                matches += 1
                                match_score = compute_match_percentage(filter_term, summary, p)
                                with st.expander(f"{i+1}. {p['title']} ({p['published']})"):
                                    st.markdown(f"**Match Percentage (filter '{filter_term}')**: {match_score}%")
                                    st.markdown("**Summary:**")
                                    st.write(summary.get("short_abstract", ""))
                                    st.markdown("**Main Focus:**")
                                    st.write(summary.get("main_focus", "(not determined)"))
                                    st.markdown("**Keywords:**")
                                    st.write(summary.get("keywords", "(none)"))
                                    st.markdown("---")
                                    with st.expander("Show full original abstract"):
                                        st.write(p['summary'])

                        if matches == 0:
                            st.info(f"No papers matched '{filter_term}'. Try another keyword like 'nlp' or 'face recognition'.")
                        st.session_state.messages.append({"role": "assistant", "content": f"Filtered papers by {filter_term}."})
                        st.stop()

                if user_input_lc == 'all':
                    indices = list(range(len(st.session_state.research_state["papers"])))
                else:
                    # Robust parsing: extract all numbers from the string
                    indices = [int(n) - 1 for n in re.findall(r'\d+', prompt)]
                
                # Validation
                if not indices:
                    st.error("I couldn't find any numbers in your message. Please enter the indices of the papers you'd like (e.g., '1, 3, 5') or 'all'.")
                elif any(i < 0 or i >= len(st.session_state.research_state["papers"]) for i in indices):
                    st.error(f"Some indices are out of range. Please pick numbers between 1 and {len(st.session_state.research_state['papers'])}.")
                else:
                    selected_papers = [st.session_state.research_state["papers"][idx] for idx in indices]
                    st.session_state.research_state["selected_indices"] = indices
                    st.session_state.research_state["selected_papers"] = selected_papers
                    
                    with st.spinner(f"Analyzing {len(selected_papers)} papers and generating 5+ page manuscript..."):
                        # Trigger the full graph execution for the selected papers
                        final_state = {
                            "research_topic": st.session_state.research_state["topic"],
                            "paper_results": st.session_state.research_state["papers"],
                            "selected_papers": selected_papers,
                            "analyses": [],
                            "final_manuscript": "",
                            "final_report_path": "",
                            "latex_report_path": "",
                            "tectonic_pdf_path": None,
                            "status": "starting"
                        }
                        result = app.invoke(final_state)
                        
                        st.session_state.research_state["final_paper_path"] = result['tectonic_pdf_path']
                        st.session_state.research_state["final_tex_path"] = result['latex_report_path']
                        
                        # Pre-load files into session state to avoid re-run issues
                        if result['tectonic_pdf_path'] and os.path.exists(result['tectonic_pdf_path']):
                            with open(result['tectonic_pdf_path'], "rb") as f:
                                st.session_state.research_state["pdf_data"] = f.read()
                        if result['latex_report_path'] and os.path.exists(result['latex_report_path']):
                            with open(result['latex_report_path'], "rb") as f:
                                st.session_state.research_state["tex_data"] = f.read()
                        
                        response = f"I have analyzed the **{len(selected_papers)}** selected papers and generated a synthesized, 5+ page research manuscript for you!"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Force a rerun to show the download buttons that are outside the chat input loop
                        st.rerun()

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                # Log the full error to the terminal for debugging
                import traceback
                traceback.print_exc()

# Final Downloads Section (Persists through re-runs)
if st.session_state.research_state["pdf_data"] is not None:
    st.divider()
    st.subheader("📥 Final Research Assets")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download Full PDF Manuscript",
            data=st.session_state.research_state["pdf_data"],
            file_name="manuscript.pdf",
            mime="application/pdf",
            key="pdf_download_persistent"
        )
    with col2:
        if st.session_state.research_state["tex_data"] is not None:
            st.download_button(
                label="📁 Download LaTeX Source",
                data=st.session_state.research_state["tex_data"],
                file_name="manuscript.tex",
                mime="text/plain",
                key="tex_download_persistent"
            )
