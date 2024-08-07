import yaml


class Config:

    def __init__(self):
        """"
        define possible parameters and initialize to None
        """

        self.model_class = None

        self.do_show_figs = None
        self.do_save_figs = None

        self.verbose = None

        self.input_data_path = None

        self.curve_strategy = None
        self.ldf_type = None
        self.product_dimension_name = None
        self.has_scenarios = None

        self.visualize = None
        self.do_export = None
        self.output_path = None

        self.is_set = False

    def set_from_yml(self, filename):
        with open(filename, 'r') as stream:
            data = yaml.safe_load(stream)
        for key, value in data.items():
            if key not in self.__dict__:
                raise ValueError(f"Parameter {key} from yml file not found in config container")
            setattr(self, key, value)
        self.is_set = True

    def __getattribute__(self, name: str):
        """
        If any config value is accessed, check if config is set first
        """
        if not object.__getattribute__(self, 'is_set') and name not in ['__dict__', 'set_from_yml']:
            raise ValueError("Config not set. Please use cfg.set_from_yml() at the beginning of your program.")
        return object.__getattribute__(self, name)


cfg = Config()
