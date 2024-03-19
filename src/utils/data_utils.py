import pandas as pd
import argparse, logging, pickle, ast, glob
from mimic_utils import filter_notes
import numpy as np
from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import train_test_split
from datetime import datetime
import math
import os
import matplotlib.pyplot as plt

def create_pr_classes_plot(pr_data):
    colors = ["forestgreen", "royalblue", "lightcoral", "lightgray", 'teal', 
              'darkviolet', 'goldenrod', 'olive', 'firebrick', 'slategray',
              'grey', 'pink', 'green', 'orange', 'red', 'dimgray', 'yellow', 'peru']
    fig, ax = plt.subplots()
    value_counts = pr_data['class'].value_counts()
    ax.bar(list(value_counts.index), value_counts.values, color=colors)
    ax.legend()
    stop = 0
def match_label_to_distribution_class(labels, borders) -> pd.DataFrame:
    DIST_CLASSES = ["short_head", "middle_body", "long_tail"]
    prev_border = 0
    class_list = []
    for idx, border in enumerate(borders[1:]):
        class_list.append([DIST_CLASSES[idx] for _ in range(border-prev_border)])
        if idx == 2:
            class_list.append(["long_tail"])
        prev_border = border
    class_list = [dist for sl in class_list for dist in sl]
    return pd.DataFrame.from_dict({"labels":labels, "dist_class": class_list})

def get_label_borders_and_percentiles(df):
    expl = df.explode("labels")
    label_count = expl.labels.value_counts()
    label_count = label_count.reset_index().rename(columns={"count":"counts"})
    label_count["percentiles"] = label_count.counts.map(lambda x: x / len(df))
    borders = []
    NUM_PERCENTILES = 3
    frequencies = label_count.counts
    frequencies = frequencies / sum(frequencies)
    tmp_sum = 0
    border_count = 0
    for rank, frequency in enumerate(sorted(frequencies, key=lambda x: x, reverse=True)):
        tmp_sum += frequency
        if tmp_sum >= (sum(frequencies) / NUM_PERCENTILES) * border_count:
            borders.append(rank)
            border_count += 1
    return borders, label_count

def load_all_splits(data_dir, task):
    if task in ['dia', 'pro']:
        test_data = pd.read_csv(data_dir + '/' + task + "/test_fold_1_simplified_true.csv", index_col="index")
        test_data.labels = test_data.labels.map(lambda x: ast.literal_eval(x))
        validation_data = pd.read_csv(data_dir + '/' + task + "/dev_fold_1_simplified_true.csv", index_col="index")
        validation_data.labels = validation_data.labels.map(lambda x: ast.literal_eval(x))
        training_data = pd.read_csv(data_dir + '/' + task + "/train_fold_1_simplified_true.csv", index_col="index")
        training_data.labels = training_data.labels.map(lambda x: ast.literal_eval(x))
        return training_data, test_data, validation_data
    if task in ['pd', 'los']:
        test_data = pd.read_csv(data_dir + task + "/test.csv").rename(
            columns={"class": "labels"}
            )
        train_data = pd.read_csv(data_dir + task + "/train.csv").rename(
            columns={"class": "labels"}
            )
        val_data = pd.read_csv(data_dir + task + "/val.csv").rename(
            columns={"class": "labels"}
            )
        return train_data, val_data, test_data

def extract_admissions_with_labels(df: pd.DataFrame, col: str = 'icd_code'):
    gb = df.drop_duplicates(['subject_id', 'hadm_id', col]).groupby(['subject_id', 'hadm_id'], dropna=False)
    admissions = gb[col].agg([lambda x: x.isna().any(), lambda x: x.str.cat(sep=',')]) \
        .rename(columns={'<lambda_0>': 'has_nan', '<lambda_1>': 'labels'})
    return admissions.reset_index(level=0, names='subject_id').drop_duplicates()

def create_all_labels(df): 
    return sorted(df.labels.str.split(',').explode().unique())
def save_labels(labels, output_path):
    with open(output_path + "/all_labels.txt", "w") as f:
        f.write(",".join(labels))
def remove_missing_labels(hosp_dataset, missing_labels):
    note_idx = []
    for label in missing_labels:
        note_idx.append(hosp_dataset[hosp_dataset.labels.map(lambda x: True if label in x else False)].index.tolist())
    note_idx = [item for sublist in note_idx for item in sublist]
    return hosp_dataset.drop(list(set(note_idx)))
    
def label_to_pos_map(all_codes):
    label_to_pos = dict([(code,pos) for code, pos in zip(sorted(all_codes),range(len(all_codes)))])
    pos_to_label = dict([(pos,code) for code, pos in zip(sorted(all_codes),range(len(all_codes)))])
    return label_to_pos, pos_to_label


def label_to_tensor(data, label_to_pos):
    tmp = np.zeros((len(data), 
                    len(label_to_pos)))
    c = 0
    for idx, row in data.iterrows():
        for code in row['labels']:
                tmp[c, label_to_pos[code]] = 1
        c += 1

    return tmp


def stratified_sampling_multilearn(df, y, train_data_output_path, simplified): 

    df = df.reset_index(drop=True).sample(frac=1, random_state=42)
    df.labels = df.labels.str.split(",")
    tmp_df = df.explode('labels')
    label_count = tmp_df.labels.value_counts()
    label_count = label_count.reset_index().rename(columns={"count":"counts"})
    label_count_df = pd.merge(tmp_df,label_count,on='labels', how='left')
    
    rare_code_patients = set(label_count_df[label_count_df.counts < 5].subject_id)
    
    df_non_rare = df[~df.subject_id.isin(rare_code_patients)].reset_index()
    df_rare = df[df.subject_id.isin(rare_code_patients)].reset_index()
    
    non_rare_code_patients_index = (~df.subject_id.isin(rare_code_patients)).values
    label_to_pos, pos_to_label = label_to_pos_map(y)
    y = label_to_tensor(df, label_to_pos)
    y_non_rare = y[non_rare_code_patients_index]
    
    k_fold = IterativeStratification(n_splits=3, order=1)
    nfold = 1
    for train, test in tqdm(k_fold.split(df_non_rare, y_non_rare)):
        
        df_train = df_non_rare.loc[train]
        train_subj_ids = df_train.subject_id
        
        df_test = df_non_rare.loc[test] 
        test_rest = df_test[df_test.subject_id.isin(train_subj_ids)]
        df_test = df_test[~df_test.subject_id.isin(train_subj_ids)]
        
        y_test = y_non_rare[df_test.index, :]
        
        df_train = pd.concat([df_train, test_rest])
        
        
        val_tmp, _, df_test_tmp, _ = iterative_train_test_split(df_test.values, y_test, test_size = 0.5)
        
        df_val = pd.DataFrame(val_tmp, columns=df_test.columns)
        val_subject_ids = df_val.subject_id
        
        df_test = pd.DataFrame(df_test_tmp, columns=df_test.columns)
        test_rest = df_test[df_test.subject_id.isin(val_subject_ids)]
        df_test = df_test[~df_test.subject_id.isin(val_subject_ids)]

        df_val = pd.concat([df_val, 
                            test_rest])

        
        
        df_train = pd.concat([df_train, df_rare])
        
        val_labels = set(df_val.labels.explode())
        train_labels = set(df_train.labels.explode())
        test_labels = set(df_test.labels.explode())
        
        assert len(val_labels.difference(train_labels)) == 0
        assert len(test_labels.difference(train_labels)) == 0
        
        assert len(df) == len(df_test) + len(df_train) + len(df_val)
        assert len(set(df_train.subject_id).intersection(set(df_test.subject_id))) == 0
        assert len(set(df_train.subject_id).intersection(set(df_val.subject_id))) == 0 
        assert len(set(df_test.subject_id).intersection(set(df_val.subject_id))) == 0
        
        
        df_train.reset_index(drop=True).to_csv(f"{train_data_output_path}/train_fold_{nfold}_simplified_{str(simplified).lower()}.csv", index=False)
        df_val.reset_index(drop=True).to_csv(f"{train_data_output_path}/dev_fold_{nfold}_simplified_{str(simplified).lower()}.csv", index=False)
        df_test.reset_index(drop=True).to_csv(f"{train_data_output_path}/test_fold_{nfold}_simplified_{str(simplified).lower()}.csv", index=False)
    
        nfold = nfold + 1
        break   
def get_all_labels(data_path): 
    with open(f'{data_path}/icd_10_all_labels_admission_mimiciv_dia.pcl', 'rb') as f: 
       labels = pickle.load(f)
    return labels

    
def remove_duplicate_text(df): 
    df['label_sizes'] = df.labels.apply(len)
    df = df.sort_values(by='label_sizes')
    df = df.drop_duplicates(subset='text', keep='last')
    df = df.drop('label_sizes', axis=1)
    return df

def merge_and_save(admissions, labels, output_path):
    merged = pd.merge(admissions, labels, left_index=True, right_index=True)
    merged.to_csv(output_path, index=True)
    return merged

def load_mimiciv_splits(data_dir, data_splits, filtered_notes, drop_icu):
    splits = {}
    for split in data_splits:
        if split == "hosp":
            admissions = pd.read_csv(data_dir + f"/{split}/admissions.csv.gz", index_col="hadm_id")
            date_format = '%Y-%m-%d %H:%M:%S'
            admissions["los"] = admissions.index.map(lambda x: (datetime.strptime(admissions.loc[x].dischtime, date_format) - datetime.strptime(admissions.loc[x].admittime, date_format)).seconds/86400)
            hosp_admission_notes = pd.merge(filtered_notes, admissions, on=["hadm_id"])
            splits[split] = hosp_admission_notes.drop_duplicates()
        elif split == "icu":
            icu_stays = pd.read_csv(data_dir + f"/{split}/icustays.csv.gz")
            icu_stays = icu_stays[~icu_stays.hadm_id.duplicated(keep=False)].set_index("hadm_id")
            icu_notes = pd.merge(icu_stays, filtered_notes, on=['hadm_id'])
            #icu_admission_notes = pd.merge(filtered_notes, icu_admissions, on=["hadm_id"])
            splits[split] = icu_notes.drop_duplicates()
        else:
            print("No correct split found. Correct split value = hosp | icu")
    #Drop ICU from Hosp split
    if drop_icu:
       splits['hosp'] = splits['hosp'].drop(splits['icu'].index) 

    return splits

def create_task_ds_by_splits(data_dir, data_splits_dict, tasks, simplify=True):
    datasets_dict = {}
    datasets_dict["simplified"] = simplify
    for split, data in data_splits_dict.items():
        datasets_dict[split] = {}
        for task in tasks:
            if task in ["dia", "pro"]:
                code_file = "/hosp/diagnoses_icd.csv.gz" if task == "dia" else '/hosp/procedures_icd.csv.gz'
                codes = pd.read_csv(data_dir + code_file, dtype={'icd_version': int, 'subject_id': int, 'icd_code': str}, usecols=('icd_version', 'subject_id', 'icd_code', 'hadm_id'))
                codes = codes[codes.icd_version == 10]
                labeled_admissions = extract_admissions_with_labels(codes)
                labeled_admission_notes = pd.merge(data, labeled_admissions.drop(columns=["subject_id"]), left_index=True, right_index=True)
                if simplify:
                    #Cut first three digits for diagnoses, first four digits for procedures
                    cut_off = 3 if task == "dia" else 4
                    #Use ''set()'' to filter duplicates after cutting
                    labeled_admission_notes.labels = labeled_admission_notes.labels.map(lambda x: ",".join(set([xs[:cut_off] for xs in x.split(",")])))
                datasets_dict[split][task] = labeled_admission_notes[['text', 'labels', "subject_id"]]
            elif task == "los":
                    data["class"] = data.los.map(hosp_los_to_class) if split == 'hosp' else data.los.map(icu_los_to_class)
                    datasets_dict[split][task] = data[["text", "class"]]
    return datasets_dict


def icu_los_to_class(x):

    if x <= 3:
        los_class = 0
    elif 3 < x <= 7:
        los_class = 1 
    elif 7 < x <= 14:
        los_class = 2
    else:
        los_class = 3
    return los_class
def create_and_save_dataset_splits(datasets_dict, output_dir):
    simplified = datasets_dict["simplified"]
    del datasets_dict["simplified"]
    for split, tasks in datasets_dict.items():
        for task, data in tasks.items():
            if task in ["dia", 'pro']:
                labels = create_all_labels(data)
                stratified_sampling_multilearn(data, labels, f'{output_dir}/{split}/{task}', simplified)
            elif task == 'los':
                x = data["text"].to_numpy()
                y = data['class'].to_numpy()
                X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                train_df = pd.DataFrame({"text" : X_train.tolist(), "class" : Y_train.tolist()}, columns=['text', 'class'])
                test_df = pd.DataFrame({"text" : X_test.tolist(), "class" : Y_test.tolist()}, columns=['text', 'class'])
                test_val_split = np.array_split(test_df, 2)
                test_df = test_val_split[0]
                val_df = test_val_split[1]
                train_df.to_csv(f'{output_dir}/{split}/{task}/train.csv', index=False)
                test_df.to_csv(f'{output_dir}/{split}/{task}/test.csv', index=False)
                val_df.to_csv(f'{output_dir}/{split}/{task}/val.csv', index=False)
def load_all_data(data_dir, splits, tasks):
    ds_dict = {}
    for split in splits:
        ds_dict[split] = {}
        for task in tasks:
            if os.path.exists(f'{data_dir}/{split}/{task}/'):
                ds_dict[split][task] = {}
                filepath = f'{data_dir}/{split}/{task}/'
                for file in glob.glob(f'{filepath}*'):
                        df = pd.read_csv(file)
                        name = file.replace(filepath, '').replace('.csv', '')
                        ds_dict[split][task][name] = df
    return ds_dict
def create_patient_routing(transfers, notes):
    admit_events = transfers[transfers.eventtype == 'admit']
    admit_events.index = admit_events.index.astype(int)
    admit_events = admit_events[['careunit']]
    unknowns = admit_events[admit_events.careunit == 'Unknown']
    admit_events = admit_events.drop(unknowns.index)
    admit_events = admit_events.careunit.map(map_careunit)
    routing_notes = notes.merge(admit_events, how="inner", on='hadm_id')
    x = routing_notes["text"].to_numpy()
    y = routing_notes['careunit'].to_numpy()
    X_train, X_test = train_test_split(routing_notes, test_size=0.3, random_state=42)
    train_df = pd.DataFrame({"text" : X_train.tolist(), "class" : X_train.tolist()}, columns=['text', 'class'], index=routing_notes.index)
    test_df = pd.DataFrame({"text" : X_test.tolist(), "class" : X_test.tolist()}, columns=['text', 'class'], index=routing_notes.index)
    test_val_split = np.array_split(test_df, 2)
    test_df = test_val_split[0]
    val_df = test_val_split[1]

    train_df.to_csv(f'/Users/toroe/Data/pr_test/train.csv')
    test_df.to_csv(f'/Users/toroe/Data/pr_test/test.csv')
    val_df.to_csv(f'/Users/toroe/Data/pr_test/val.csv')


def map_careunit(careunit):
    surgery = ['Med/Surg', 'Surgery', 'Medical/Surgical (Gynecology)','Med/Surg/GYN', 'Surgery/Pancreatic/Biliary/Bariatric', 'Surgery/Trauma', 'Med/Surg/Trauma','Cardiology Surgery Intermediate', 'Cardiac Surgery', 'Thoracic Surgery', 'PACU']
    labor_delivery = ['Labor & Delivery', 'Obstetrics (Postpartum & Antepartum)', 'Obstetrics Postpartum' ,'Obstetrics Antepartum']
    cardiology = [ 'Cardiology', 'Medicine/Cardiology Intermediate', 'Medicine/Cardiology']
    oncology = ['Hematology/Oncology', 'Hematology/Oncology Intermediate']
    observation = ['Emergency Department Observation', 'Observation']
    neurology = ['Neurology', 'Neuro Intermediate','Neuro Stepdown']

    if careunit in surgery:
        careunit = "surgery"
    elif careunit in labor_delivery:
        careunit = 'obstetrics'
    elif careunit in cardiology:
        careunit = 'cardiology'
    elif careunit in oncology:
        careunit = 'oncology'
    elif careunit in observation:
        careunit = 'observation'
    elif careunit in neurology:
        careunit = 'neurology'
    else:
        careunit = careunit.lower()
    return careunit
def create_mp_data(data_dict):
    for module, data in data_dict.items():
        data = data.drop(data[data.text.str.contains('patient passed away')].index)
        data = data.drop(data[data.text.str.contains('patient expired')].index)
        data = data.drop(data[data.text.str.contains('patient deceased')].index)
        data = data.drop(data[data.text.str.contains('patient died')].index)
        data = data.drop(data[data.text.str.contains('pronounced dead')].index)
        x = data["text"].to_numpy()
        y = data['hospital_expire_flag'].to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        train_df = pd.DataFrame({"text" : X_train.tolist(), "class" : Y_train.tolist()}, columns=['text', 'class'])
        test_df = pd.DataFrame({"text" : X_test.tolist(), "class" : Y_test.tolist()}, columns=['text', 'class'])
        test_val_split = np.array_split(test_df, 2)
        test_df = test_val_split[0]
        val_df = test_val_split[1]
        




    


