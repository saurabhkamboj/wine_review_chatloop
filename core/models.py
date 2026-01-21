from pydantic import BaseModel, Field
from typing import Optional


class QueryClassification(BaseModel):
    type: str = Field(
        description=(
            "If the query mentions attributes like country, province, variety, or description, set type='semantic'. "
            "If it only specifies filters like price, points, or a tasters' name, set type='keyword'. "
            "If both are present, use 'semantic'."
        )
    )
    taster_name: Optional[str] = Field(
        default=None,
        description='The name of the taster (null if not mentioned).'
    )
    min_points: Optional[int] = Field(
        default=None,
        description='The minimum points that the wine should have (null if not mentioned).'
    )
    max_points: Optional[int] = Field(
        default=None,
        description='The maximum points that the wine should have (null if not mentioned).'
    )
    min_price: Optional[float] = Field(
        default=None,
        description='The minimum price of the wine (null if not mentioned).'
    )
    max_price: Optional[float] = Field(
        default=None,
        description='The maximum price of the wine (null if not mentioned).'
    )
