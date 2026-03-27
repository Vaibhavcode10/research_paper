import streamlit as st
from graph import app
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
        "ideas": [],
        "final_paper_path": None,
        "final_tex_path": None,
        "pdf_data": None,
        "tex_data": None
    }

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
                
                response = f"I found these 10 recent papers on **{prompt}**. Please choose which ones you'd like me to analyze (e.g., enter '1, 3, 5, 6, 7, 8' to pick 6, or 'all' for all 10):\n\n"
                for i, p in enumerate(result["paper_results"]):
                    response += f"{i+1}. **{p['title']}** ({p['published']})\n   *Summary:* {p['summary'][:200]}...\n\n"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        # 2. If topic is decided but papers are not selected
        elif not st.session_state.research_state["selected_indices"]:
            try:
                import re
                if prompt.lower().strip() == 'all':
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
