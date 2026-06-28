import os

import h5py
from pathlib import Path
from typing import Optional, Sequence, Union
import pandas as pd
import numpy as np
from fastai.vision.learner import load_learner
from FeaturesbasedMIL.mil.data import make_dataset
from FeaturesbasedMIL.recorded_information import get_attention_score
import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.all import (
    Learner, DataLoader)

PathLike = Union[str, Path]

def get_target_enc(mil_learn):
    return mil_learn.dls.train.dataset._datasets[-1].encode

def get_FeabasedMIL_coords_scores(h5_feature_path: Path, model: nn.Module):

    feats, coords, sizes = [], [], []
    with h5py.File(h5_feature_path, 'r') as f:
        feats.append(torch.from_numpy(f['feats'][:]).float())
        sizes.append(len(f['feats']))
        coords.append(torch.from_numpy(f['coords'][:]))
    feats, coords = torch.cat(feats), torch.cat(coords)

    encoder = model.encoder.eval()
    attention = model.attention.eval()
    head = model.head.eval()

    # calculate attention, scores etc.
    encs = encoder(feats)
    patient_atts = torch.softmax(attention(encs), dim=0).detach()
    patient_scores = torch.softmax(head(encs), dim=1).detach()
    normed_patient_atts = (patient_atts-patient_atts.min())/(patient_atts.max()-patient_atts.min())
    patient_weighted_scores = normed_patient_atts*patient_scores

    scores = patient_atts.numpy()
    scores -= scores.min()
    scores /= (scores.max() - scores.min())
    normed_patient_atts_np = normed_patient_atts.numpy()


    FeabasedMIL_coords_scores = coords.numpy(), scores

    return FeabasedMIL_coords_scores

def get_deploy_cohort_df(feature_dir: Union[Path, str],target_label: str) -> pd.DataFrame:
    #clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(clini_table, dtype=str)
    #slide_df = pd.read_csv(slide_csv, dtype=str)
    #df = clini_df.merge(slide_df, on='PATIENT')

    # remove uninteresting
    #df = df[df[target_label].isin(categories)]
    # remove slides we don't have
    h5s = set(feature_dir.glob('*.h5'))
    assert h5s, f'no features found in {feature_dir}!'
    h5_df = pd.DataFrame(h5s, columns=['slide_path'])
    h5_df['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem)
    h5_df['PATIENT'] = h5_df.slide_path.map(lambda p: p.stem)
    # h5_df[target_label] = 'nonMSIH'#'0'#'nonMSIH'
    h5_df[target_label] = '0'  # '0'#'nonMSIH'
    len = h5_df.shape[0]
    if len > 1:
        # h5_df[target_label][len-1] = 'MSIH'#'1'#'MSIH'
        h5_df[target_label][len - 1] = '1'  # '1'#'MSIH'
        df = h5_df
    elif len==1:
        h5_df2 = pd.DataFrame(h5s, columns=['slide_path'])
        h5_df2['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem)
        h5_df2['PATIENT'] = 'copy_for_Predict'
        # h5_df2[target_label] = 'MSIH'#'1'#'MSIH'
        h5_df2[target_label] = '1'  # '1'#'MSIH'
        ndf = pd.concat([h5_df,h5_df2])
        df = ndf
        #h5_df = pd.concat(h5_df,h5_df2)

    #df = df.merge(h5_df, on='FILENAME')

    # reduce to one row per patient with list of slides in `df['slide_path']`
    patient_df = df.groupby('PATIENT').first().drop(columns='slide_path')
    patient_slides = df.groupby('PATIENT').slide_path.apply(list)
    df = patient_df.merge(patient_slides, left_on='PATIENT', right_index=True).reset_index()

    return df

def FeabasedMIL_deploy(
    test_df: pd.DataFrame, learn: Learner, *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None, cont_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'
    if target_label is None: target_label = learn.target_label
    if cat_labels is None: cat_labels = learn.cat_labels
    if cont_labels is None: cont_labels = learn.cont_labels

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
    print(f"FeabasedMIL_deploy categories is {categories}")
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))


    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=1)#os.cpu_count())
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    for index, row in test_df.iterrows():
        save_path = Path(str(row["slide_path"][0])[:-3] + ".csv")
        attention_score = get_attention_score(row["slide_path"][0], learn.model)
        att_score_df = attention_score
        att_score_df.to_csv(save_path, index=False)



    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        target_label: test_df[target_label].values,
        **{f'{target_label}_{cat}': patient_preds[:, i]
            for i, cat in enumerate(categories)}})

    # calculate loss
    patient_preds = patient_preds_df[[
        f'{target_label}_{cat}' for cat in categories]].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1))
    patient_preds_df['loss'] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs),
        reduction='none')

    patient_preds_df['pred'] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[[
        'PATIENT',
        target_label,
        'pred',
        *(f'{target_label}_{cat}' for cat in categories),
        'loss']]
    patient_preds_df = patient_preds_df.sort_values(by='loss')
    patient_preds_df = patient_preds_df.drop(target_label, axis=1) #删除指定列
    patient_preds_df = patient_preds_df.drop('loss', axis=1)  # 删除指定列
    num_clo = patient_preds_df[patient_preds_df['PATIENT'] =='copy_for_Predict']
    patient_preds_df = patient_preds_df.drop(num_clo.index)

    return patient_preds_df

def deploy_FeabasedMIL_categorical_model_(
    feature_dir: PathLike,
    model_path: PathLike,
    output_path: PathLike,
    *,
    target_label: Optional[str] = None,
    cat_labels: Optional[str] = None,
    cont_labels: Optional[str] = None,
) -> None:
    """Deploy a categorical model on a cohort's tile's features.

    Args:
        clini_excel:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        target_label:  Label to train for.
        feature_dir:  Path containing the features.
        model_path:  Path of the model to deploy.
        output_path:  File to save model in.
    """
    for file in os.listdir(Path(feature_dir)):
        deal_features_dir = Path(feature_dir/file)
        out_features_dir = Path(output_path/file)

        model_path = Path(model_path)
        if (preds_csv := out_features_dir/'patient-preds.csv').exists():
            print(f'{preds_csv} already exists!  Skipping...')
            return

        learn = load_learner(model_path)
        target_enc = get_target_enc(learn)

        categories = target_enc.categories_[0]

        target_label = target_label or learn.target_label

        test_df = get_deploy_cohort_df(deal_features_dir,target_label)
        patient_preds_df = FeabasedMIL_deploy(
            test_df=test_df, learn=learn, target_label=target_label)
        out_features_dir.mkdir(parents=True, exist_ok=True)
        patient_preds_df.to_csv(preds_csv, encoding='utf-8-sig', index=False)

def svae_pedict_info(predict_df, output_path, model_path):
    df_colums = predict_df.columns.to_list()
    patient_num = len(predict_df["PATIENT"].to_list())

    for num in range(patient_num):
        patient_ID = predict_df[df_colums[0]][num]
        with open(output_path / (str(patient_ID) + '_MSI.txt'), 'w') as f:
            f.write(str(predict_df[df_colums[0]][num]) + '\n')
            f.write(str(model_path) + '\n')
            f.write(str(predict_df[df_colums[2]][num]) + '\n')
            MSIstatue = get_msistatue(predict_df[df_colums[2]][num])
            f.write(str(MSIstatue) + '\n')

def get_msistatue(MSIscore):
    if MSIscore >=0.5:
        MSIstatue = 'MSI-U'
    else:
        MSIstatue = 'MSS'
    return MSIstatue



