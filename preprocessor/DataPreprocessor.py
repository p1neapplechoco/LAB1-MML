from .DataFiller import DataFiller
from .DataScaler import DataScaler
from .DataEncoder import DataEncoder

class DataPreprocessor(DataFiller, DataEncoder, DataScaler):
    def __init__(self, df):
        DataFiller.__init__(self, df)
        DataEncoder.__init__(self, df)
        DataScaler.__init__(self, df)