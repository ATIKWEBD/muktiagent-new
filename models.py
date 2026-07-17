from typing import List, Optional, Annotated, TypedDict
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl
import operator

# ==========================================
# 1. PLATFORM & CATEGORY ENUMS
# ==========================================
class PlatformTarget(str, Enum):
    AMAZON = "amazon"
    FLIPKART = "flipkart"
    MYNTRA = "myntra"

class DiscountType(str, Enum):
    PERCENTAGE = "percentage"
    FLAT = "flat"
    CASHBACK = "cashback"

# ==========================================
# 2. DATA EXTRACTION SCHEMAS (Gemini Output)
# ==========================================
class BankOffer(BaseModel):
    bank_name: str = Field(
        description="The name of the bank providing the promotional offer (e.g., HDFC, SBI, ICICI, Axis). If it applies to any bank's credit card, output 'Any Bank'."
    )
    discount_type: DiscountType = Field(
        description="Whether the discount is an exact flat amount off, a percentage reduction, or a post-purchase cashback program."
    )
    discount_value: float = Field(
        description="The numerical value of the discount itself. For example: for '10% instant discount', output 10.0. For 'Save Flat ₹2000', output 2000.0."
    )
    max_discount_cap: Optional[float] = Field(
        default=None, 
        description="The maximum absolute upper limit of the discount if capped. E.g., '10% off up to ₹1,500', output 1500.0. If unlimited, leave empty."
    )
    condition: Optional[str] = Field(
        default=None, 
        description="Any operational constraints or spending thresholds required to trigger the offer. E.g., 'On Credit Card EMI transactions only' or 'Minimum order value of ₹5,000'."
    )

class ProductOffer(BaseModel):
    product_id: str = Field(
        description="The unique identifier or stock-keeping unit parsed from the page or URL. For Amazon, extract the ASIN (e.g., B0CHX1W1XY). For Flipkart, extract the FSN or PID."
    )
    product_name: str = Field(
        description="The full, high-fidelity descriptive title of the product, including variant attributes like color and storage capacity if listed."
    )
    base_price: float = Field(
        description="The manufacturer's maximum retail price (MRP) or original strike-through listed price before any reductions. Strip out commas or currency symbols."
    )
    selling_price: float = Field(
        description="The final current active price listed on the storefront before bank card promotions are applied."
    )
    is_in_stock: bool = Field(
        description="True if the item is currently available for purchase. False if the page displays text like 'Out of stock', 'Currently unavailable', or 'Sold Out'."
    )
    bank_offers: List[BankOffer] = Field(
        default_factory=list, 
        description="An exhaustive array of all bank card, EMI, wallet, or platform specific instant deduction offers mapped to this single product listing."
    )

# ==========================================
# 3. INTER-AGENT CENTRAL WORKFLOW STATE
# ==========================================
class ExtractionGraphState(TypedDict):
    """
    Manages the ledger of shared data and operational checkpoints 
    as a specific scraping job executes across the multi-agent graph.
    """
    job_id: str
    url: str
    platform: PlatformTarget
    
    # Browser Agent Handshakes
    raw_markdown: Optional[str]
    screenshot_base64: Optional[str]
    requires_vision_fallback: bool
    
    # Extractor Agent Structured Payloads
    # Annotated with operator.add enables automatic item appending across graph state mutations
    extracted_offers: Annotated[List[ProductOffer], operator.add]
    
    # Validator Agent Deduplication Ledger
    deduped_hashes: Annotated[List[str], operator.add]
    
    # System Telemetry & Resiliency Flags
    errors: Annotated[List[str], operator.add]
    retry_count: int
    is_completed: bool