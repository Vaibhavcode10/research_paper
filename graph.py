from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, START, END
from tools.arxiv_tool import search_arxiv, download_pdf
from tools.pdf_reader import extract_text_from_pdf
from tools.report_generator import create_research_report
from tools.latex_generator import generate_latex_manuscript
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import concurrent.futures

# Load Environment Variables
load_dotenv()

# Resolve API key from supported environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise EnvironmentError(
        "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment (e.g. in .env)."
    )

# Initialize the model with the requested version
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=gemini_api_key,
    max_output_tokens=25000
)

# Define the Agent State
class AgentState(TypedDict):
    research_topic: str
    paper_results: List[Dict[str, Any]]
    selected_papers: List[Dict[str, Any]]
    analyses: List[Dict[str, Any]]
    final_manuscript: str  # New field for the final synthesized text
    final_report_path: str
    latex_report_path: str
    tectonic_pdf_path: str
    status: str

# 1. Search Papers Node
def search_papers_node(state: AgentState):
    """
    Search ArXiv for papers matching the research topic.
    """
    query = state['research_topic']
    papers = search_arxiv(query, max_results=10)
    return {
        "paper_results": papers,
        "status": "papers_searched"
    }

# 2. Paper Selection Node (LLM-based)
def select_papers_node(state: AgentState):
    """
    Use Gemini to decide which 6 papers are most relevant from the search results.
    """
    papers = state['paper_results']
    topic = state['research_topic']
    
    # If the user already provided indices in the UI flow, we use those.
    # But if not, we can let the AI pick the best 6.
    if state.get('selected_papers') and len(state['selected_papers']) > 0:
        return {"status": "papers_already_selected"}

    paper_list_str = ""
    for i, p in enumerate(papers):
        paper_list_str += f"{i+1}. Title: {p['title']}\nSummary: {p['summary'][:300]}...\n\n"

    selection_prompt = ChatPromptTemplate.from_template("""
    You are an expert research curator. Here is a list of papers found for the topic: {topic}.
    Your goal is to select exactly 6 papers that provide the most comprehensive and high-quality coverage of the topic.
    
    Papers:
    {paper_list}
    
    Return ONLY a comma-separated list of the indices (1-indexed) of the 6 best papers.
    Example output: 1, 3, 5, 6, 7, 9
    """)
    
    chain = selection_prompt | llm
    response = chain.invoke({"topic": topic, "paper_list": paper_list_str})
    
    try:
        # Basic parsing of the comma-separated output
        content = response.content.strip()
        # Handle cases where LLM might add text like "Indices: 1, 2, ..."
        if ":" in content:
            content = content.split(":")[-1].strip()
        
        indices = [int(i.strip()) - 1 for i in content.split(',') if i.strip().isdigit()]
        selected = [papers[i] for i in indices if 0 <= i < len(papers)]
        
        # Ensure we don't return 0 papers
        if not selected:
             selected = papers[:6]
    except Exception as e:
        print(f"Selection failed: {e}")
        selected = papers[:6]

    return {
        "selected_papers": selected[:6],
        "status": "papers_selected_by_ai"
    }

# 3. Paper Extraction Node (Summarization)
def analyze_papers_node(state: AgentState):
    """
    Download PDFs, extract text, and extract key features from each paper.
    """
    papers = state['selected_papers']
    topic = state['research_topic']
    
    # Read the extraction prompt template
    try:
        with open("prompts/extraction_prompt.txt", "r") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        prompt_template = "Extract key insights from this paper regarding {research_topic}: {paper_content}"

    def process_single_paper(paper):
        try:
            # 1. Download PDF
            safe_title = "".join([c if c.isalnum() else "_" for c in paper['title']])[:50]
            pdf_filename = f"{safe_title}.pdf"
            pdf_path = download_pdf(paper['pdf_url'], pdf_filename)
            
            # 2. Extract Text
            content = extract_text_from_pdf(pdf_path)
            # Use a larger context for better analysis
            content_snippet = content[:20000] 
            
            # 3. LLM Extraction
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm
            
            response = chain.invoke({
                "paper_content": content_snippet,
                "research_topic": topic
            })
            extraction_text = response.content
            
            return {
                "title": paper['title'],
                "authors": paper['authors'],
                "insights": extraction_text
            }
        except Exception as e:
            print(f"Error processing {paper['title']}: {e}")
            return None

    analyses = []
    import time
    for paper in papers:
        # Respect the Gemini Free Tier limit (5 RPM)
        # 13s ensures we stay under 5 requests per minute
        print(f"Processing: {paper['title']}...")
        result = process_single_paper(paper)
        if result:
            analyses.append(result)
        time.sleep(13)
        
    return {
        "analyses": analyses,
        "status": "papers_summarized"
    }

# 4. Synthesis Node (Full Manuscript Generation)
def synthesize_manuscript_node(state: AgentState):
    """
    Synthesize all individual paper analyses into a single cohesive manuscript.
    """
    analyses = state['analyses']
    topic = state['research_topic']
    
    if not analyses:
        return {"final_manuscript": "No analyses available.", "status": "synthesis_failed"}

    # Combine analyses into a single text block for the prompt
    analyses_text = ""
    for i, a in enumerate(analyses):
        analyses_text += f"\n--- PAPER {i+1}: {a['title']} ---\n{a['insights']}\n"

    try:
        with open("prompts/synthesis_prompt.txt", "r") as f:
            synthesis_prompt_template = f.read()
    except FileNotFoundError:
        synthesis_prompt_template = "Synthesize these analyses into a full research paper on {research_topic}: {analyses_text}"

    prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
    chain = prompt | llm
    
    # Use a bigger context/longer timeout for the final generation
    response = chain.invoke({
        "analyses_text": analyses_text,
        "research_topic": topic
    })
    
    return {
        "final_manuscript": response.content,
        "status": "manuscript_synthesized"
    }

# 5. Generate Report Node
def generate_report_node(state: AgentState):
    """
    Compile insights into a PDF report and LaTeX manuscript.
    """
    analyses = state['analyses']
    final_text = state['final_manuscript']
    topic = state['research_topic']
    
    # For the simple report, we use the individual analyses
    report_path = create_research_report(analyses, topic)
    
    # For the LaTeX manuscript, we use the synthesized final text
    latex_path, tectonic_pdf_path = generate_latex_manuscript(final_text, topic)
    
    return {
        "final_report_path": report_path,
        "latex_report_path": latex_path,
        "tectonic_pdf_path": tectonic_pdf_path,
        "status": "report_generated"
    }


# Build the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("search_papers", search_papers_node)
workflow.add_node("select_papers", select_papers_node)
workflow.add_node("analyze_papers", analyze_papers_node)
workflow.add_node("synthesize_manuscript", synthesize_manuscript_node)
workflow.add_node("generate_report", generate_report_node)

# Add Edges
workflow.add_edge(START, "search_papers")
workflow.add_edge("search_papers", "select_papers")
workflow.add_edge("select_papers", "analyze_papers")
workflow.add_edge("analyze_papers", "synthesize_manuscript")
workflow.add_edge("synthesize_manuscript", "generate_report")
workflow.add_edge("generate_report", END)

# Compile
app = workflow.compile()
