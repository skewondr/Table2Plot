from utils import Dataset
import pandas as pd
from table2line import Table2Line

dataset = Dataset(data_dir='./', data_name='KorWikiTQ') 
for i, queries_tables in enumerate(dataset.train_samples):
    table = pd.DataFrame.from_records(queries_tables["table"][1:], columns=queries_tables["table"][0]).astype(str)
    Table2Line(table, i)
    # if i == 0: break
    if i == 0: break