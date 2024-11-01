import sys
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator
from typing import Dict
from abc import abstractmethod
from .named_dim_arrays import Parameter
from .dimensions import DimensionSet

class TradeModule(PydanticBaseModel):
    """ A TradeModule handles the storing and calculation of trade data for a given MFASystem."""

    dims : DimensionSet
    trade_data : Dict[str, Parameter]

    @abstractmethod
    def balance_historic_trade(self):
        pass

    @abstractmethod
    def balance_future_trade(self):
        pass

    @abstractmethod
    def predict(self):
        pass


    @classmethod
    def balance_trade(self, imports, exports, by:str):
        assert 'r' in imports.dims.letters and 'r' in exports.dims.letters, "Imports and exports must have a Region dimension."

        if by == 'maximum':
            return self.balance_trade_by_extrenum(imports, exports, 'maximum')
        elif by == 'minimum':
            return self.balance_trade_by_extrenum(imports, exports, 'minimum')
        elif by == 'scaling_average':
            return self.balance_trade_by_scaling_average(imports, exports)
        else:
            raise ValueError(f"Method {by} not recognized. Must be one of 'maximum', 'minimum', or 'scaling_average'.")


    @classmethod
    def balance_trade_by_extrenum(self, imports:Parameter, exports:Parameter, extrenum:str):
        global_imports = imports.sum_nda_over('r')
        global_exports = exports.sum_nda_over('r')

        if extrenum == 'maximum':
            max_trade = global_imports.maximum(global_exports)
        elif extrenum == 'minimum':
            max_trade = global_imports.minimum(global_exports)
        else:
            raise ValueError(f"Extrenum {extrenum} not recognized. Must be one of 'maximum' or 'minimum'.")

        import_factor = max_trade / global_imports.maximum(sys.float_info.epsilon)
        export_factor = max_trade / global_exports.maximum(sys.float_info.epsilon)

        new_imports = imports * import_factor
        new_exports = exports * export_factor

        return new_imports, new_exports


    @classmethod
    def balance_trade_by_scaling_average(cls, imports:Parameter, exports:Parameter):
        """Balances trade by scaling imports and exports by the same factor as proposed by Pehl et al."""
        net_trade = imports - exports
        absolute_net_trade = net_trade.abs()
        global_net_trade = absolute_net_trade.sum_nda_over('r')
        global_absolute_net_trade = absolute_net_trade.sum_nda_over('r')

        new_net_trade = net_trade * (1 - net_trade.sign() * global_net_trade / global_absolute_net_trade)

        new_imports = new_net_trade.maximum(0)
        new_exports = new_net_trade.minimum(0).abs()

        return new_imports, new_exports

    def __getitem__(self, key):
        if key not in self.trade_data:
            raise KeyError(f"Key {key} not found in Trade Module data.")
        return self.trade_data[key]

    def __setitem__(self, key, value):
        self.trade_data[key] = value