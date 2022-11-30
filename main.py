from utils import Dataset, save_bbox
import pandas as pd
from table2line import Table2Line
from collections import Counter

dataset = Dataset(data_dir='./', data_name='KorWikiTQ') 

str_answer = []
num_answer = []
error_header = []

def get_statics(dataset):
    for i, queries_tables in enumerate(dataset.train_samples):
        try:
            table = pd.DataFrame.from_records(queries_tables["table"][1:], columns=queries_tables["table"][0]).astype(str)
        except:
            error_header.append(i)
            continue
        try:
            float(queries_tables["answer"])
            num_answer.append(i)
        except:
            str_answer.append(i)

    for i, queries_tables in enumerate(dataset.dev_samples):
        try:
            table = pd.DataFrame.from_records(queries_tables["table"][1:], columns=queries_tables["table"][0]).astype(str)
        except:
            error_header.append(i)
            continue
        try:
            float(queries_tables["answer"])
            num_answer.append(i)
        except:
            str_answer.append(i)

    print(f'# str_answer: {len(str_answer)}, # float_answer: {len(num_answer)}, # error table: {len(error_header)}')
    print("total samples:", len(str_answer)+ len(num_answer)+ len(error_header), len(dataset.dev_samples)+len(dataset.train_samples))
    return 

cnt = []
plot_list = dict()
for i, queries_tables in enumerate(dataset.train_samples):
    try:
        table = pd.DataFrame.from_records(queries_tables["table"][1:], columns=queries_tables["table"][0]).astype(str)
    except:
        continue
    val, _ = Table2Line(table, i, queries_tables["answer"])
    cnt.append(val)
    if val == 4:
        fig_name, bbfig_name, bbox_list = _
        plot_list[i]=(fig_name, bbfig_name, bbox_list)
        # break 
    # if i>300: break
save_bbox(plot_list, "train")

print("total possible line plots:", Counter(cnt))

plot_list = dict()
for i, queries_tables in enumerate(dataset.dev_samples):
    try:
        table = pd.DataFrame.from_records(queries_tables["table"][1:], columns=queries_tables["table"][0]).astype(str)
    except:
        continue
    val = Table2Line(table, i, queries_tables["answer"])
    cnt.append(val)
    if val == 4: 
        fig_name, bbfig_name, bbox_list = _
        plot_list[i]=(fig_name, bbfig_name, bbox_list)
        # break
save_bbox(plot_list, "dev")


print("total possible line plots:", Counter(cnt))