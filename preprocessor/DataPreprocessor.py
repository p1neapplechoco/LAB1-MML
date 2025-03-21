from DataFiller import DataFiller
from DataEncoder import DataEncoder
from DataScaler import DataScaler

class DataFiller(DataFiller, DataEncoder, DataScaler):
    def __init__(self, df):
        DataFiller.__init__(self, df)
        DataEncoder.__init__(self, df)
        DataScaler.__init__(self, df)