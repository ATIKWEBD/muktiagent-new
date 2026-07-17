import asyncio
import hashlib
import os
from typing import Optional
from playwright.async_api import async_playwright
import html2text
from dotenv import load_dotenv

# Fixed Import Path for newer versions of langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Import the state and models we built in Phase 1
from models import ExtractionGraphState, ProductOffer, PlatformTarget

load_dotenv()

# ==========================================
# 1. BROWSER AGENT (DOM to Markdown)
# ==========================================
async def fetch_and_clean_page(url: str) -> Optional[str]:
    """
    Spins up a headless browser, loads the page, and converts the DOM to Markdown.
    Waits for the network to idle so dynamic JS offers load.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            # Wait until network is idle to ensure dynamic pricing & offers load
            await page.goto(url, wait_until="networkidle", timeout=15000)
            
            raw_html = await page.content()
            await browser.close()
            
            # Convert to clean Markdown
            text_maker = html2text.HTML2Text()
            text_maker.ignore_links = True
            text_maker.ignore_images = True
            text_maker.bypass_tables = False
            
            return text_maker.handle(raw_html)
            
    except Exception as e:
        print(f"Browser Agent Error: {e}")
        return None

# ==========================================
# 2. EXTRACTOR AGENT (Gemini Semantic Parsing)
# ==========================================
async def extract_offers_from_markdown(markdown_text: str) -> ProductOffer:
    """
    Passes the markdown to Gemini and enforces the ProductOffer Pydantic schema.
    """
    # Initialize Gemini
    llm = ChatGoogleGenAI(
        model="gemini-1.5-flash", 
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Bind the Pydantic schema to force Gemini to output structured JSON
    structured_llm = llm.with_structured_output(ProductOffer)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data extraction agent. Read the provided e-commerce webpage text and extract the exact pricing and bank offer details perfectly. If information is missing, do not guess. Ensure the product_id is extracted from the provided text or deduce it logically if possible."),
        ("human", "Here is the webpage content:\n\n{webpage_text}")
    ])
    
    chain = prompt | structured_llm
    
    # Execute the chain
    result = await chain.ainvoke({"webpage_text": markdown_text})
    return result

# ==========================================
# 3. VALIDATOR AGENT (Deduplication)
# ==========================================
def generate_offer_hash(platform: PlatformTarget, offer: ProductOffer) -> str:
    """
    Creates a unique SHA-256 hash based on Platform + Product ID + Price.
    """
    unique_string = f"{platform.value}_{offer.product_id}_{offer.selling_price}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()