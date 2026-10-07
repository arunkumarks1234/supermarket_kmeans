from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EvaluateRequest(BaseModel):
    features: List[str] = Field(..., description="List of features to use for clustering")
    k_min: int = Field(2, ge=2, description="Minimum number of clusters")
    k_max: int = Field(8, le=15, description="Maximum number of clusters")

class FitRequest(BaseModel):
    features: List[str] = Field(..., description="List of features to use for clustering")
    k: int = Field(..., ge=2, description="Number of clusters to fit")

class PredictRequest(BaseModel):
    unit_price: float = Field(..., ge=0, description="Unit price of the product")
    quantity: int = Field(..., ge=1, description="Quantity purchased")
    rating: float = Field(..., ge=0, le=10, description="Customer rating")
