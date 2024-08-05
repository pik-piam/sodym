from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class DimensionDefinition(BaseModel):
    name: str = Field(..., min_length=2)
    dim_letter: str = Field(..., min_length=1, max_length=1)
    filename: Optional[str] = None
    dtype: type


class SomethingWithDimensions(BaseModel):
    dim_letters: tuple

    @field_validator("dim_letters", mode='before')
    def check_dimensions(cls, v):
        for letter in v:
            if (not isinstance(letter, str)) or (len(letter) != 1):
                raise ValueError('flows must be defined using single digit dimension letters')
        return v


class FlowDefinition(SomethingWithDimensions):
    from_process: str
    to_process: str


class StockDefinition(SomethingWithDimensions):
    name: Optional[str] = None
    process: Optional[str] = None


class ParameterDefinition(SomethingWithDimensions):
    name: str


class MFADefinition(BaseModel):
    """
    All the information needed to define an MFA system.
    """
    dimensions: List[DimensionDefinition]
    processes: List[str]
    flows: List[FlowDefinition]
    stocks: List[StockDefinition]
    parameters: List[ParameterDefinition]
    scalar_parameters: Optional[list] = []
