import asyncio
from typing import Literal
from langgraph.graph import StateGraph, START, END

# Import the shared state and agents
from models import ExtractionGraphState
from agents import fetch_and_clean_page, extract_offers_from_markdown, generate_offer_hash

# ==========================================
# 1. GRAPH NODES (The Agent Workers)
# ==========================================

async def browser_node(state: ExtractionGraphState) -> dict:
    """Node 1: Navigates to the e-commerce URL and extracts the DOM to Markdown."""
    job_id = state.get("job_id", "unknown_job")
    print(f"[{job_id}] 🌐 Browser Agent running on {state['url']}...")
    
    markdown_content = await fetch_and_clean_page(state['url'])
    
    if not markdown_content:
        return {"errors": ["Browser failed to load page or extract content."]}
        
    return {"raw_markdown": markdown_content}


async def extractor_node(state: ExtractionGraphState) -> dict:
    """Node 2: Feeds the cleaned Markdown to Gemini to extract structured JSON."""
    job_id = state.get("job_id", "unknown_job")
    print(f"[{job_id}] 🧠 Extractor Agent (Gemini) processing markdown...")
    
    markdown = state.get('raw_markdown')
    if not markdown:
        return {"errors": ["No markdown available for extraction."]}
        
    try:
        extracted_offer = await extract_offers_from_markdown(markdown)
        # Wrap in list to append to the state ledger via operator.add
        return {"extracted_offers": [extracted_offer]}
    except Exception as e:
        return {"errors": [f"Gemini Extraction Failed: {str(e)}"]}


def validator_node(state: ExtractionGraphState) -> dict:
    """Node 3: Validates and hashes the extracted data to prevent duplicates."""
    job_id = state.get("job_id", "unknown_job")
    print(f"[{job_id}] 🛡️ Validator Agent generating hashes...")
    
    offers = state.get('extracted_offers', [])
    if not offers:
        return {"errors": ["No offers to validate."]}
        
    # Grab the most recently extracted offer
    latest_offer = offers[-1]
    
    # Generate deduplication hash
    offer_hash = generate_offer_hash(state['platform'], latest_offer)
    print(f"[{job_id}] Hash generated: {offer_hash}")
    
    return {
        "deduped_hashes": [offer_hash],
        "is_completed": True
    }

# ==========================================
# 2. CONDITIONAL ROUTING (The Control Logic)
# ==========================================

def route_after_browser(state: ExtractionGraphState) -> Literal["extractor_node", "__end__"]:
    """Decides where to go after the browser attempts to load the page."""
    if state.get("errors") and len(state.get("errors")) > 0:
        print(f"[{state.get('job_id')}] Routing to END due to browser errors.")
        return END
    return "extractor_node"

def route_after_extractor(state: ExtractionGraphState) -> Literal["validator_node", "__end__"]:
    """Decides where to go after Gemini attempts extraction."""
    if state.get("errors") and len(state.get("errors")) > 0:
         print(f"[{state.get('job_id')}] Routing to END due to extraction errors.")
         return END
    return "validator_node"

# ==========================================
# 3. BUILD AND COMPILE THE GRAPH
# ==========================================

def build_offer_discovery_graph():
    """Constructs and compiles the LangGraph state machine."""
    workflow = StateGraph(ExtractionGraphState)
    
    # Register Nodes
    workflow.add_node("browser_node", browser_node)
    workflow.add_node("extractor_node", extractor_node)
    workflow.add_node("validator_node", validator_node)
    
    # Set Entry Point
    workflow.add_edge(START, "browser_node")
    
    # Add Conditional Edges
    workflow.add_conditional_edges("browser_node", route_after_browser)
    workflow.add_conditional_edges("extractor_node", route_after_extractor)
    
    # Standard Exit
    workflow.add_edge("validator_node", END)
    
    return workflow.compile()

# ==========================================
# 4. STANDALONE LOCAL TESTING
# ==========================================
async def test_run():
    graph = build_offer_discovery_graph()
    initial_state = {
        "job_id": "job_local_test_001",
        "url": "https://www.amazon.in/Apple-iPhone-15-128-GB/dp/B0CHX1W1XY",
        "platform": "amazon",
        "retry_count": 0,
        "is_completed": False,
        "errors": [],
        "extracted_offers": [],
        "deduped_hashes": []
    }
    
    print("\n--- Starting Local Orchestrator Pipeline ---\n")
    final_state = await graph.ainvoke(initial_state)
    print("\n--- Pipeline Finished. Final State Snapshot ---")
    
    if final_state.get('extracted_offers'):
         print("\n✅ Extracted Data JSON:")
         print(final_state['extracted_offers'][0].model_dump_json(indent=2))
    
    if final_state.get('errors'):
        print(f"\n❌ Errors Encountered: {final_state['errors']}")

if __name__ == "__main__":
    asyncio.run(test_run())