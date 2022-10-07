'''
Tools for modelling chip probabilities
'''

import os, json
import warnings
import pandas
import geopandas
import numpy as np
from time import time
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

import statsmodels.api as sm
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
)
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import esda
import pointpats
from libpysal import weights

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
w_class_names = ['w_'+i for i in class_names]


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
    tab, 
    cols,
    w=None,
    criterium=weights.Queen, # Needs to have a `from_dataframe` method
    crit_kwds={},
    prefix='w_',
    flag_islands=True,
    add_card=True
):
    if w is None:
        w_user = criterium.from_dataframe(
            tab, ids=range(len(tab)), **crit_kwds
        ) # NOTE: positional index
        w_k1 = weights.KNN.from_dataframe(tab, k=1, ids=range(len(tab)))
        w = weights.w_union(w_user, w_k1)
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

def path2chipsize_arch(p):
    pieces = (
        os.path.split(p)[1]
        .split('.')[0]
        .replace('v2_', '')
        .split('_')
    )
    chipsize = int(pieces[0])
    # Arch
    if 'multi' in pieces:
        arch = 'mor'
    elif 'slided' in pieces:
        arch = 'sic'
    else:
        arch = 'bic'
    return chipsize, arch

def premodelling_process(p, cw):
    chipsize, arch = path2chipsize_arch(p)
    db = geopandas.read_parquet(p+'_labels.parquet')
    db.index = pandas.RangeIndex(len(db))
    # Labels
    if arch == 'mor':
        type_ids = db.drop(columns=['geometry', 'split']).columns
        db['signature_type'] = (db
                                [type_ids]
                                .idxmax(axis=1)
                               )
        outliers = db['signature_type'].isin(
            ['9_3', '9_6', '9_7', '9_8']
        )
        db = db.drop(columns=type_ids)#.loc[~outliers, :]
    db['label'] = pandas.Categorical(db['signature_type'].map(cw))
    # Probs
    columns = [ # Provided by MF on Aug. 17th
        "Urbanity", 
        "Dense residential neighbourhoods",
        "Connected residential neighbourhoods",
        "Dense urban neighbourhoods",
        "Accessible suburbia",
        "Open sprawl",
        "Warehouse_Park land",
        "Gridded residential quarters",
        "Disconnected suburbia",
        "Countryside agriculture", 
        "Wild countryside", 
        "Urban buffer"
    ]
    columns = [i.lower().replace(' ', '_') for i in columns]
    probs = pandas.DataFrame(
        np.load(p+'_prediction.npy'), columns=columns
    )[class_names] # Re-order following hierarchy
    # Remove outliers after join
    if arch == 'mor':
        db = db[~outliers]
    # Keep only data in ML split
    db = db.join(probs).query('(split == "ml_train") | (split == "ml_val")')
    # Spatial Lag
    sp_lag = db.groupby('split').apply(
        lag_columns, 
        cols=class_names, 
        criterium=weights.DistanceBand, 
        crit_kwds={'threshold': chipsize * 10 * 1.5} 
    )
    db = db.join(sp_lag.drop(columns=['island', 'card']))
    return db, f'{chipsize}_{arch}'

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
        logite[c] = sm.Logit(endog_c, exog).fit(disp=False)
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

def model_runner(f, params, verbose, fo=None):
    log = ''
    try:
        res = f(*params)
        log += logger(f'{datetime.now()} |Log| {f} completed successfully\n', verbose, fo)
    except:
        res = None
        log += logger(f'{datetime.now()} |Log| {f} failed\n', verbose, fo)
    return log

def logger(log, verbose=False, fo=None):
    if verbose:
        print(log)
    if fo is not None:
        with open(fo, 'a') as l:
            l.write(log)
    return log

def run_all_models(
    db, prefix, out_folder, verbose=False, fo=None, models=None, ignoreWarnings=True
):
    if models is None:
        models = ['maxprob', 'logite', 'gbt']
    train_ids = db.query('split == "ml_train"').index
    val_ids = db.query('split == "ml_val"').index
    log = logger(f'\t### {prefix} ###\n', verbose, fo)
    if 'maxprob' in models:
        log += model_runner(     # Maxprob
            run_maxprob, (
                class_names, 'label', db, train_ids, val_ids, prefix, out_folder
            ), verbose=verbose, fo=fo
        )
    if 'logite' in models:
        log += model_runner(     # LogitE baseline
            run_logite, (
                class_names, 'label', db, train_ids, val_ids, f'baseline_{prefix}', out_folder
            ), verbose=verbose, fo=fo
        )
        log += model_runner(     # LogitE baseline + wx
            run_logite, (
                class_names + w_class_names, 
                'label', 
                db, 
                train_ids, 
                val_ids, 
                f'baseline-wx_{prefix}', 
                out_folder
            ), verbose=verbose, fo=fo
        )
    if 'gbt' in models:
                             # GBT
        hbgb_param_grid = {
            'max_iter': [50, 100, 150, 200, 300],
            'learning_rate': [0.01, 0.05] + np.linspace(0, 1, 11)[1:].tolist(),
            'max_depth': [5, 10, 20, 30, None],
        }
        '''
        hbgb_param_grid = {
            'max_iter': [50],
            'learning_rate': [0.01, 0.05],
            'max_depth': [30, None],
        }
        '''
                                 # Grid baseline
        grid = GridSearchCV(
            HistGradientBoostingClassifier(),
            hbgb_param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1
        )
        log += model_runner(
            grid.fit, (
                db.query('split == "ml_train"')[class_names],
                db.query('split == "ml_train"')['label']
            ), verbose, fo=fo
        )
        log += model_runner(     # HBGBT baseline
            run_tree, (
                class_names,
                'label', 
                db,
                train_ids,
                val_ids,
                HistGradientBoostingClassifier(**grid.best_params_),
                f'baseline_{prefix}',
                out_folder
            ), verbose=verbose, fo=fo
        )
                                 # Grid baseline + wx
        grid = GridSearchCV(
            HistGradientBoostingClassifier(),
            hbgb_param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1
        )
        log += model_runner(
            grid.fit, (
                db.query('split == "ml_train"')[class_names + w_class_names],
                db.query('split == "ml_train"')['label']
            ), verbose, fo=fo
        )
        log += model_runner(     # HBGBT baseline + wx
            run_tree, (
                class_names + w_class_names,
                'label', 
                db,
                train_ids,
                val_ids,
                HistGradientBoostingClassifier(**grid.best_params_),
                f'baseline-wx_{prefix}',
                out_folder
            ), verbose=verbose, fo=fo
        )
    return log
    
                    #################
                    #  Performance  #
                    #################

csa_order = [
    f'{cs}_{a}' for cs in [8, 16, 32, 64] for a in ['bic', 'sic', 'mor']
]

models_order = [
    'maxprob', 
    'logite_baseline', 
    'logite_baseline-wx', 
    'HistGradientBoostingClassifier_baseline', 
    'HistGradientBoostingClassifier_baseline-wx'
]
                
def parse_path(p):
    _, f = os.path.split(p)
    pieces = f.replace('_y_pred.pq', '').split('_')
    arch = pieces[-1]
    chipsize = pieces[-2]
    algo = pieces[0]
    if len(pieces) == 4:
        variant = pieces[1]
    else:
        variant = ''
    return algo, variant, chipsize, arch

def compile_chipsize_arch(chipsize, arch, ps):
    fs = [p for p in ps if parse_path(p)[2:] == (str(chipsize), arch)]
    db = pandas.concat([(
        pandas.read_parquet(f)
        [['y_pred']]
        .rename(columns={'y_pred': '_'.join(parse_path(f)[:2]).strip('_')})
    ) for f in fs], axis=1)
    geo = geopandas.read_parquet(
        f'{os.path.split(ps[0])[0]}/geo_labels_{chipsize}_{arch}.parquet'
    ).assign(chipsize_arch=f'{chipsize}_{arch}')
    return geo.join(db)
    
def build_perf(
    y_true_train, 
    y_pred_train, 
    y_true_val,
    y_pred_val,
    class_names,
    meta={},
    preds_path=None
):
    '''
    Compute performance scores for a set of predictions and labels, with a train/val split
    ...
    
    Arguments
    ---------
    y_true_train : pandas.Series
                   Train set of labels
    y_pred_train : pandas.Series
                   Train set of predictions
    y_true_val : pandas.Series
                 Validation set of labels
    y_pred_val : pandas.Series
                 Validation set of predictions
    
    Returns
    -------
    meta : dict
           Set of scores as a dict
    '''
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

def build_cm_plot(cm, maxcount=None, std=False, ax=None, cbar=True):
    cm = pandas.DataFrame(
        cm, class_names, class_names
    )
    if ax is None:
        f, ax = plt.subplots(1)
    if std is not False:
        cm = cm.div(cm.sum(axis=1), axis='index')
        maxcount = 1
    sns.heatmap(
        cm, vmin=0, vmax=maxcount, cmap='viridis', ax=ax, cbar=cbar
    )
    ax.set_axis_off()
    return ax
                    #################
                    #    Spatial    #
                    #################
            
def sp_process_csa(params):
    '''
    Compute all spatial metrics for a geography
    ...
    
    Arguments
    ---------
    params : tuple
             Parameters
                tab : GeoDataFrame
                      Geo-Table with columns:
                          - `geometry`
                          - `<model_name1>`
                          - `<model_name2>`
                          - ...
                csa : str
                      f'{chipsize}_{arch}'
                models : None/list
                         Models to evaluate spatially (if `None`, all are used)
                label : str
                        Column for true labels

    Returns
    -------
    sp_res : DataFrame
             Table with all spatial scores
    '''
    tab, csa, models, label = params
    if models is None:
        models = tab.drop(columns=['geometry', label]).columns.tolist()
    if type(models) is str:
        models = [models]
    '''
    # Build Ws
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        w_queen = weights.Queen.from_dataframe(tab, ids=range(len(tab)))
        w_k1 = weights.KNN.from_dataframe(tab, k=1, ids=range(len(tab)))
        w_union = weights.w_union(w_queen, w_k1)
        w_union.transform = 'r'
        #
        min_thr = weights.min_threshold_distance(
            np.array([tab.centroid.x, tab.centroid.y]).T
        )
        w_thr = weights.DistanceBand.from_dataframe(tab, min_thr)
    wd = {'union': w_union, 'thr': w_thr}
    '''
    # Coords
    xys = tab.centroid
    xys = np.array([xys.x, xys.y]).T
    # Spatial stats
    sp_res = []
    for model in models+[label]:
        sp_res.append(spatial_scores((
            tab[model], None, xys, f'{csa}_{model}'
        )))
    return pandas.concat(sp_res)
            
def spatial_scores(params):
    y, wd, xys, name = params
    support = 5
    sp_res = {
        f'ripley_{s}_d{i}': {} for s in ['k', 'g'] for i in range(support)
    }
    if wd is not None:
        for w in wd:
            sp_res[f'moran_{w}'] = {}
            sp_res[f'jc_{w}'] = {}
    for c in y.unique():
        b = (y == c).astype(float)
        c_xys = xys[(y == c).values, :]
        if wd is not None:
            # Moran & JC
            if len(b.unique()) > 1:
                for w in wd:
                    # Moran
                    sp_res[f'moran_{w}'][c] = esda.Moran(b, wd[w], permutations=1).I
                    # Join counts
                    wd[w].transform = 'O'
                    sp_res[f'jc_{w}'][c] = esda.Join_Counts(
                        b, wd[w], permutations=1
                    ).bb
                # Quadrat
                #sp_res['quadrat'][c] = pointpats.QStatistic(
                #   c_xys, shape='hexagon', lh=10#000
                #).chi2
            else:
                for w in wd:
                    sp_res[f'moran_{w}'][c] = None
                    sp_res[f'jc_{w}'][c] = None
                    #sp_res['quadrat'][c] = None
        # Ripley's G
        try:
            stat = pointpats.g_test(
                c_xys, support=support, n_simulations=1
            )
            for i in range(support):
                sp_res[f'ripley_g_d{i}'][c] = stat.statistic[i]
        except:
            for i in range(support):
                sp_res[f'ripley_g_d{i}'][c] = None
        # Ripley's K
        try:
            stat = pointpats.k_test(
                c_xys, support=support, n_simulations=1
            )
            for i in range(support):
                sp_res[f'ripley_k_d{i}'][c] = stat.statistic[i]
        except:
            for i in range(support):
                sp_res[f'ripley_k_d{i}'][c] = None
    pieces = name.split('_')
    cs = pieces[0]
    arch = pieces[1]
    model = '_'.join(pieces[2:])
    sp_res = pandas.DataFrame(sp_res).stack().reset_index().rename(
        columns={'level_0': 'signature', 'level_1': 'metric', 0: 'value'}
    ).assign(csa=f'{cs}_{arch}').assign(model=model)
    return sp_res

def score_distance(tab):
    out_tab = tab.query('model != "label"')
    try:
        label_value = tab.query('model == "label"')['value'].iloc[0]
        out_tab['value'] = abs(
            (out_tab['value'] - label_value) * 100 / (label_value + 1e-10)
        )
    except:
        out_tab['value'] = pandas.NA
    return out_tab[['model', 'value']].rename(columns={'value': 'dist'})
