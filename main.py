from utils import Dataset, save_pickle, load_pickle
import pandas as pd
from table2line import Table2Line
from collections import Counter
from tqdm import tqdm 
from random import sample

dataset = Dataset(data_dir='./', data_name='KorWikiTQ') 

str_answer = []
num_answer = []
error_header = []

def get_statics(dataset):
    for i, queries_tables in enumerate(tqdm(dataset.train_samples, ncols=50)):
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

    for i, queries_tables in enumerate(tqdm(dataset.dev_samples, ncols=50)):
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
plot_list = []

idx = 0
for queries_tables in tqdm(dataset.train_samples, ncols=50):
    if idx > 100:
        break
    try:
        table = pd.DataFrame.from_records(queries_tables["table"][1:], columns=queries_tables["table"][0]).astype(str)
    except:
        continue
    # index = i*4
    # for j in range(4):
    proxy_n = sample([0, 1, 2, 3], 1)[0]
    val, fig_name_bbox = Table2Line(table, idx, proxy_n, queries_tables["question"],  queries_tables["answer"])
    cnt.append(val)
    if val == 0:
        fig_name, _, bbox_list = fig_name_bbox
        # continue
        plot_list.append(
            {
                "qid": queries_tables["qid"],
                "question": queries_tables["question"],
                "answer": queries_tables["answer"],
                "image": fig_name.split('/')[-1],
                "bboxes": bbox_list,
            }
        )
        idx += 1
    # if i > 1200 : break 

for queries_tables in tqdm(dataset.dev_samples, ncols=50):
    if idx > 100:
        break
    try:
        table = pd.DataFrame.from_records(queries_tables["table"][1:], columns=queries_tables["table"][0]).astype(str)
    except:
        continue
    proxy_n = sample([0, 1, 2, 3], 1)[0]
    val, fig_name_bbox = Table2Line(table, idx, proxy_n, queries_tables["question"],  queries_tables["answer"])
    cnt.append(val)
    if val == 0: 
        fig_name, _, bbox_list = fig_name_bbox
        plot_list.append(
            {
                "qid": queries_tables["qid"],
                "question": queries_tables["question"],
                "answer": queries_tables["answer"],
                "image": fig_name.split('/')[-1],
                "bboxes": bbox_list,
            }
        )
        idx += 1
        # break

save_pickle(plot_list, "line_annotation.pkl")
# print(load_pickle("KorWikiTQ/line_annotation.pkl"))
print("total possible line plots:", Counter(cnt))
