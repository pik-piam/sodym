from pydantic import BaseModel as PydanticBaseModel, Field, field_validator
from typing import List, Optional


class DimensionDefinition(PydanticBaseModel):
    name: str = Field(..., min_length=2)
    dim_letter: str = Field(..., min_length=1, max_length=1)
    filename: Optional[str] = None
    dtype: type


class DefinitionWithDimLetters(PydanticBaseModel):
    dim_letters: tuple

    @field_validator("dim_letters", mode='before')
    def check_dimensions(cls, v):
        for letter in v:
            if (not isinstance(letter, str)) or (len(letter) != 1):
                raise ValueError('Dimensions must be defined using single digit letters')
        return v


class FlowDefinition(DefinitionWithDimLetters):
    from_process: str
    to_process: str


class StockDefinition(DefinitionWithDimLetters):
    name: Optional[str] = None
    process: Optional[str] = None


class ParameterDefinition(DefinitionWithDimLetters):
    name: str


class MFADefinition(PydanticBaseModel):
    """
    All the information needed to define an MFA system.
    """
    dimensions: List[DimensionDefinition]
    processes: List[str]
    flows: List[FlowDefinition]
    stocks: List[StockDefinition]
    parameters: List[ParameterDefinition]
    scalar_parameters: Optional[list] = []
