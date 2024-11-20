import sys
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator
from abc import abstractmethod
from .named_dim_arrays import Parameter
from typing import Optional, Callable

class Trade(PydanticBaseModel):
    """ A TradeModule handles the storing and calculation of trade data for a given MFASystem."""

    imports : Parameter
    exports : Parameter
    balancer : Optional[Callable] = None
    predictor : Optional[Callable] = None


    @model_validator(mode='after')
    def validate_region_dimension(self):
        assert 'r' in self.imports.dims.letters, "Imports must have a Region dimension."
        assert 'r' in self.exports.dims.letters, "Exports must have a Region dimension."

        return self

    @model_validator(mode='after')
    def validate_trade_dimensions(self):
        assert self.imports.dims == self.exports.dims, "Imports and Exports must have the same dimensions."

        return self


    def balance(self, **kwargs):
        if self.balancer is not None:
            self.balancer(self, **kwargs)
        else:
            raise NotImplementedError("No balancer function has been implemented for this Trade object.")

    def predict(self):
        if self.predictor is not None:
            assert 'h' in self.imports.dims.letters and 'h' in self.exports.dims.letters, \
                "Trade data must have a historic time dimension."
            self.predictor()
        else:
            raise NotImplementedError("No predictor function has been implemented for this Trade object.")

    def __getitem__(self, key):
        if key == 'Imports':
            return self.imports
        elif key == 'Exports':
            return self.exports
        else:
            raise KeyError(f"Key {key} not found in Trade data - has to be either 'Imports' or 'Exports'.")

    def __setitem__(self, key, value):
        if key == 'Imports':
            self.imports = value
        elif key == 'Exports':
            self.exports = value
        else:
            raise KeyError(f"Key {key} has to be either 'Imports' or 'Exports'.")

    # balancing class methods

    @classmethod
    def balance_by_extrenum(cls, trade, by:str):
        global_imports = trade.imports.sum_nda_over('r')
        global_exports = trade.exports.sum_nda_over('r')

        if by == 'maximum':
            reference_trade = global_imports.maximum(global_exports)
        elif by == 'minimum':
            reference_trade = global_imports.minimum(global_exports)
        elif by == 'imports':
            reference_trade = global_imports
        elif by == 'exports':
            reference_trade = global_exports
        else:
            raise ValueError(f"Extrenum {by} not recognized. Must be one of "
                             f"'maximum', 'minimum', 'imports' or 'exports'.")

        import_factor = reference_trade / global_imports.maximum(sys.float_info.epsilon)
        export_factor = reference_trade / global_exports.maximum(sys.float_info.epsilon)

        trade.imports = trade.imports * import_factor
        trade.exports = trade.exports * export_factor

    @classmethod
    def balance_by_scaling(cls, trade):
        net_trade = trade.imports - trade.exports
        global_net_trade = net_trade.sum_nda_over('r')
        global_absolute_net_trade = net_trade.abs().sum_nda_over('r')

        # avoid division by zero, net_trade will be zero when global absolute net trade is zero anyways
        global_absolute_net_trade = global_absolute_net_trade.maximum(sys.float_info.epsilon)

        new_net_trade = net_trade * (1 - net_trade.sign() * global_net_trade / global_absolute_net_trade)

        trade.imports = new_net_trade.maximum(0)
        trade.exports = new_net_trade.minimum(0).abs()