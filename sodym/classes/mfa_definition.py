import numpy as np
from pydantic import (
    BaseModel as PydanticBaseModel,
    AliasChoices,
    Field,
    field_validator,
    model_validator
)
from typing import List, Optional


class DimensionDefinition(PydanticBaseModel):
    name: str = Field(..., min_length=2)
    letter: str = Field(..., min_length=1, max_length=1, validation_alias=AliasChoices("letter", "dim_letter"))
    dtype: type


class DefinitionWithDimLetters(PydanticBaseModel):
    dim_letters: tuple

    @field_validator("dim_letters", mode="before")
    def check_dimensions(cls, v):
        for letter in v:
            if (not isinstance(letter, str)) or (len(letter) != 1):
                raise ValueError("Dimensions must be defined using single digit letters")
        return v


class FlowDefinition(DefinitionWithDimLetters):
    from_process_name: str = Field(validation_alias=AliasChoices("from_process_name", "from_process"))
    to_process_name: str = Field(validation_alias=AliasChoices("to_process_name", "to_process"))


class StockDefinition(DefinitionWithDimLetters):
    name: str = "undefined stock"
    process_name: str = Field(default="undefined process", validation_alias=AliasChoices("process", "process_name"))


class ParameterDefinition(DefinitionWithDimLetters):
    name: str


class MFADefinition(PydanticBaseModel):
    """All the information needed to define an MFA system."""

    dimensions: List[DimensionDefinition]
    processes: List[str]
    flows: List[FlowDefinition]
    stocks: List[StockDefinition]
    parameters: List[ParameterDefinition]
    scalar_parameters: Optional[list] = []

    @model_validator(mode='after')
    def check_dimension_letters(self):
        defined_dim_letters = [dd.letter for dd in self.dimensions]
        for item in self.flows + self.stocks + self.parameters:
            correct_dims = [letter in defined_dim_letters for letter in item.dim_letters]
            if not np.all(correct_dims):
                raise ValueError(f'Undefined dimension in {item}')
        return self
