'''
Tools for modelling chip probabilities
'''

import os, json
import pandas
import numpy as np
from libpysal import weights
from time import time

import statsmodels.api as sm
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
)
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

TRAIN_FRAC = 0.7
                    #################
                    ### Wrangling ###
                    #################
    
class_names = [
    'urbanity',
    'dense_urban_neighbourhoods',
    'dense_residential_neighbourhoods',
    'connected_residential_neighbourhoods',
    'gridded_residential_quarters',
    'accessible_suburbia',
    'disconnected_suburbia',
    'open_sprawl',
    'warehouse_park_land',
    'urban_buffer',
    'countryside_agriculture',
    'wild_countryside'
]

def parse_nn_json(json_p):
    with open(json_p) as f:
        d = json.load(f)
        cd2nm = {}
        for codes, name in zip(d['meta_class_map'], d['meta_class_names']):
            for code in codes:
                cd2nm[f'signature_type_{code}'] = name.lower().replace(
                    " ", "_"
                ).replace("/", "_")
        cd2nm = pandas.Series(cd2nm)
        return cd2nm
    
def lag_columns(
    tab, cols, w=None, prefix='w_', flag_islands=True, add_card=True
):
    if w is None:
        w = weights.Queen.from_dataframe(tab) # NOTE: positional index
        w.transform = 'R'
    col_names = [prefix+i for i in cols]
    w_tab = pandas.DataFrame(
        np.zeros(tab[cols].shape), index=tab.index, columns=col_names
    )
    for col in cols:
        w_tab[prefix+col] = weights.lag_spatial(w, tab[col])
    if flag_islands is True:
        w_tab['island'] = False
        w_tab.iloc[w.islands, -1] = True # Assumes positional index
    if add_card is True:
        w_tab['card'] = pandas.Series(w.cardinalities).sort_index().values
    return w_tab

def build_prob_wprob_corrs(
    class_names, w_class_names, db_all, lag_all
):
    corrs = pandas.DataFrame(
        np.zeros((len(class_names), len(w_class_names))),
        index=class_names,
        columns=w_class_names
    )
    for v in class_names:
        for wv in w_class_names:
            corrs.loc[v, wv] = db_all[v].corr(lag_all[wv])
    return corrs

def interact_vars(db, vars_a, vars_b, auto_drop=False):
    a_by_b = pandas.DataFrame(
        np.zeros((db.shape[0], len(vars_a)*len(vars_b))),
        db.index
    )
    i = 0
    for va in vars_a:
        for vb in vars_b:
            a_by_b[i] = db[va] * db[vb]
            a_by_b = a_by_b.rename(columns={i:f'{va}-x-{vb}'})
            i+=1
    return a_by_b

                    #################
                    ### Modelling ###
                    #################

def run_tree(
    x_name, 
    y_name, 
    db, 
    train_ids, 
    val_ids, 
    model,
    model_name_xtra=None, 
    res_path=None
):
    # Train
    X_train = db.loc[train_ids, x_name]
    t0 = time()
    model.fit(X_train, db.loc[train_ids, y_name])
    t1 = time()
    rf_pred_train = pandas.Series(model.predict(X_train), train_ids)
    # Validation
    X_val = db.loc[val_ids, x_name]
    rf_pred_val = pandas.Series(model.predict(X_val), val_ids)
    # Results
    model_name = build_model_name(type(model).__name__, model_name_xtra)
    rf_res = build_perf(
        db.loc[train_ids, y_name], 
        rf_pred_train, 
        db.loc[val_ids, y_name], 
        rf_pred_val, 
        class_names,
        {
            'model_name': model_name,
            'meta_runtime': t1-t0,
            'model_params': model.get_params()
        },
        res_path
    )
    write_json(rf_res, res_path)
    return rf_res

def run_maxprob(
    x_name, y_name, db, train_ids, val_ids, model_name_xtra=None, res_path=None
):
    t0 = time()
    mp_pred = db[x_name].idxmax(axis=1)
    t1 = time()
    model_name = build_model_name('maxprob', model_name_xtra)
    mp_res = build_perf(
        db.loc[train_ids, y_name], 
        mp_pred.loc[train_ids], 
        db.loc[val_ids, y_name], 
        mp_pred.loc[val_ids], 
        x_name, 
        {
            'model_name': model_name,
            'meta_runtime': t1-t0,
        },
        res_path
    )
    write_json(mp_res, res_path)
    return mp_res

def logite_fit(endog, exog, classes):
    logite = {}
    for c in classes:
        endog_c = pandas.Series(np.zeros(endog.shape), index=endog.index)
        endog_c[endog == c] = 1
        logite[c] = sm.Logit(endog_c, exog).fit()
    return logite

def logite_predict(logite, exog, classes):
    pred_probs = pandas.DataFrame(
        np.zeros((exog.shape[0], len(classes))), exog.index, classes
    )
    for c in classes:
        pred_probs[c] = logite[c].predict(exog)
    return pred_probs

passer = lambda df: df

def run_logite(
    x_name, 
    y_name, 
    db, 
    train_ids, 
    val_ids, 
    model_name_xtra=None, 
    res_path=None, 
    scale_x=True, 
    log_x=True,
    interact=None, # (list, list)
    class_names=None
):
    if class_names is None:
        class_names = db[y_name].unique().tolist()
    scaler = logger = lambda df: df
    scaler = scale if scale_x is True else passer
    logger = np.log1p if log_x is True else passer
    # Train
    X_train = pandas.DataFrame(
        scaler(logger(db.loc[train_ids, x_name])), train_ids, x_name
    )
    if interact is not None:
        X_train = X_train.join(interact_vars(X_train, *interact))
    t0 = time()
    logite = logite_fit(db.loc[train_ids, y_name], X_train, class_names)
    t1 = time()
    logite_pred_train = logite_predict(logite, X_train, class_names).idxmax(axis=1)
    # Validation
    X_val = pandas.DataFrame(
        scaler(logger(db.loc[val_ids, x_name])), val_ids, x_name
    )
    if interact is not None:
        X_vals = X_vals.join(interact_vars(X_val, *interact))
    logite_pred_val = logite_predict(logite, X_val, class_names).idxmax(axis=1)
    # Results
    model_name = build_model_name('logite', model_name_xtra)
    logite_res = build_perf(
        db.loc[train_ids, y_name], 
        logite_pred_train, 
        db.loc[val_ids, y_name], 
        logite_pred_val, 
        class_names,
        {
            'model_name': model_name,
            'meta_runtime': t1-t0,
        },
        res_path
    )
    logite_res['model_params'] = {
        'coefs': [], 'scale_x': scale_x, 'log_x': log_x
    }
    for c in class_names:
        logite_res['model_params']['coefs'].append(logite[c].params.to_dict())
    write_json(logite_res, res_path)
    return logite_res

def run_mlogit(
    x_name, y_name, db, train_ids, val_ids, model_name_xtra=None, res_path=None
):
    # Train
    X_train = pandas.DataFrame(
        scale(db.loc[train_ids, x_name]), train_ids, x_name
    )
    t0 = time()
    mlogit_mod = sm.MNLogit(db.loc[train_ids, y_name], X_train)
    mlogit_res = mlogit_mod.fit()
    t1 = time()
    # Train prediction
    mlogit_pred_train = mlogit_res.predict(X_train).rename(
        columns=mlogit_mod._ynames_map
    ).idxmax(axis=1)
    # Validation
    X_val = pandas.DataFrame(
        scale(np.log1p(db.loc[val_ids, x_name])), val_ids, x_name
    )
    mlogit_pred_val = mlogit_res.predict(X_val).rename(
        columns=mlogit_mod._ynames_map
    ).idxmax(axis=1)
    # Results
    model_name = build_model_name('mlogit', model_name_xtra)
    mlogit_res = build_perf(
        db.loc[train_ids, y_name], 
        mlogit_pred_train, 
        db.loc[val_ids, y_name], 
        mlogit_pred_val, 
        [mlogit_mod._ynames_map[i] for i in range(mlogit_mod.J)],
        {
            'model_name': model_name,
            'meta_runtime': t1-t0,
        }
    )
    write_json(mlogit_res, res_path)
    return mlogit_res
    
                    #################
                    #  Performance  #
                    #################
    
def build_perf(
    y_true_train, 
    y_pred_train, 
    y_true_val,
    y_pred_val,
    class_names,
    meta={},
    preds_path=None
):
    splits = (y_true_train, y_pred_train), (y_true_val, y_pred_val)
    t_vc = y_true_train.value_counts()
    v_vc = y_true_val.value_counts()
    meta['meta_trainval_counts'] = []
    for c in class_names:
        meta['meta_trainval_counts'].append([int(t_vc[c]), int(v_vc[c])])
    for (y_true, y_pred), subset in zip(splits, ['train', 'val']):
        meta[f'perf_model_accuracy_{subset}'] = accuracy_score(y_true, y_pred)
        meta[f'perf_f1_{subset}'] = f1_score(
            y_true, y_pred, average=None
        ).tolist()
        meta[f'perf_macro_f1_w_{subset}'] = f1_score(
            y_true, y_pred, average='weighted'
        )       
        meta[f'perf_macro_f1_avg_{subset}'] = f1_score(
            y_true, y_pred, average='macro'
        )
        meta[f'perf_confusion_{subset}'] = confusion_matrix(
            y_true, y_pred, labels=class_names
        ).tolist()
        meta[f'perf_kappa_{subset}'] = cohen_kappa_score(y_true, y_pred)
        meta[f'perf_within_class_accuracy_{subset}'] = []
        for c in class_names:
            cids = y_true == c
            meta[f'perf_within_class_accuracy_{subset}'].append(
                accuracy_score(y_true[cids], y_pred[cids])
            )
    if 'meta_class_names' not in meta:
        meta['meta_class_names'] = class_names
    if 'meta_n_class' not in meta:
        meta['meta_n_class'] = len(class_names)
    if preds_path is not None:
        pandas.concat([
            pandas.DataFrame({
                'id': y_pred_train.index, 'y_pred': y_pred_train, 'Validation': False
            }),
            pandas.DataFrame({
                'id': y_pred_val.index, 'y_pred': y_pred_val, 'Validation': True
            })
        ]).to_parquet(
            os.path.join(preds_path, meta['model_name'].replace(' ', '_') + '_y_pred.pq')
        )
        meta['meta_preds_path'] = preds_path
    return meta

def write_json(res, res_path):
    if res_path is not None:
        with open(
            os.path.join(res_path, res['model_name'].replace(' ', '_') + '.json'), 'w'
        ) as fo:
            json.dump(res, fo, indent=4)
        return None
    return None

def build_model_name(base_name, model_name_xtra):
    model_name = base_name
    if model_name_xtra is not None:
        model_name += '_' + model_name_xtra
    return model_name

def build_cm_plot(cm, maxcount=None, std=False, ax=None):
    cm = pandas.DataFrame(
        cm, class_names, class_names
    )
    if ax is None:
        f, ax = plt.subplots(1)
    if std is not False:
        cm = cm.div(cm.sum(axis=1), axis='index')
        maxcount = 1
    sns.heatmap(
        cm, vmin=0, vmax=maxcount, cmap='viridis', ax=ax
    )
    ax.set_axis_off()
    return ax
