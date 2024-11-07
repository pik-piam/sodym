import sys
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator
from abc import abstractmethod
from .named_dim_arrays import Parameter
from .dimensions import DimensionSet

class Trade(PydanticBaseModel):
    """ A TradeModule handles the storing and calculation of trade data for a given MFASystem."""

    dims : DimensionSet
    imports : Parameter
    exports : Parameter

    @model_validator(mode='after')
    def validate_region_dimension(self):
        assert 'r' in self.imports.dims.letters, "Imports must have a Region dimension."
        assert 'r' in self.exports.dims.letters, "Exports must have a Region dimension."

        return self

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

class BalancedTrade(Trade):

    @abstractmethod
    def balance(self, **kwargs):
        pass

class MinMaxBalancedTrade(BalancedTrade):

    def balance(self, by:str):
        global_imports = self.imports.sum_nda_over('r')
        global_exports = self.exports.sum_nda_over('r')

        if by == 'maximum':
            max_trade = global_imports.maximum(global_exports)
        elif by == 'minimum':
            max_trade = global_imports.minimum(global_exports)
        else:
            raise ValueError(f"Extrenum {by} not recognized. Must be one of 'maximum' or 'minimum'.")

        import_factor = max_trade / global_imports.maximum(sys.float_info.epsilon)
        export_factor = max_trade / global_exports.maximum(sys.float_info.epsilon)

        self.imports = self.imports * import_factor
        self.exports = self.exports * export_factor


class ScalingBalancedTrade(BalancedTrade):

    def balance(self):
        """Balances trade by scaling imports and exports by the same factor as proposed by Pehl et al."""
        net_trade = self.imports - self.exports
        absolute_net_trade = net_trade.abs()
        global_net_trade = net_trade.sum_nda_over('r')
        global_absolute_net_trade = absolute_net_trade.sum_nda_over('r')

        # avoid division by zero, net_trade will be zero when global absolute net trade is zero anyways
        global_absolute_net_trade = global_absolute_net_trade.maximum(sys.float_info.epsilon)

        new_net_trade = net_trade * (1 - net_trade.sign() * global_net_trade / global_absolute_net_trade)

        self.imports = new_net_trade.maximum(0)
        self.exports = new_net_trade.minimum(0).abs()