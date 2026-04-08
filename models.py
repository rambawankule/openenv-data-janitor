from pydantic import BaseModel
from typing import List, Optional, Any

class Action(BaseModel):
    command: str  # e.g., "impute_mean", "fix_dates", "drop_duplicates", "submit"
    column: Optional[str] = None

class Observation(BaseModel):
    data_preview: str  # A string representation of df.head()
    missing_counts: str
    status_message: str

class Reward(BaseModel):
    value: float  # 0.0 to 1.0
