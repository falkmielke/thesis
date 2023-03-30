#!/usr/bin/env python3

import os as OS
import pickle as PCK
import numpy as NP
import pandas as PD
import matplotlib as MP
import matplotlib.pyplot as PLT
import seaborn as SB
n_prediction_samples = 16384 # 1024 # 16384

def FormatTable(table):

    lines = table.split('\n')
    n_rows = len(lines)
    n_cols = len(lines[0])
    hline = lines[1]

    # split header
    header = lines[0].split('|')
    header = [hd[:-1] + (' ' if '/' in hd else '/') for hd in header ] # make sure there is a slash
    PadToLength = lambda txt, l: txt + (' '*(l-len(txt)) if len(txt) < l else '')
    head0 = '|'.join([PadToLength(hd.split('/')[0], len(hd)) for hd in header])
    head1 = '| '.join([PadToLength(hd.split('/')[1], len(hd)-1) for hd in header])

    # add hlines
    lines = [hline] + [head0[1:], head1] + lines[1:-1] + [hline] + [lines[-1]] + [hline]

    # return table
    return '\n'.join(lines)


def GetPredictionData(force_reload = False):

    storage_file = f'./predictions/age_predictions_{n_prediction_samples}_individuals_preloaded.pck'

    if OS.path.exists(storage_file) and not force_reload:
        print ('loading pre-stored prediction diffs....')
        with open(storage_file, 'rb') as storage:
            data = PCK.load(storage)

    else:
        print ('re-calculating individual prediction diffs')
        data = PD.read_csv(f'./predictions/age_predictions_{n_prediction_samples}.csv', sep = ';')
        data.set_index(['cycle_idx', 'sample_nr'], inplace = True)
        data.index.names = ['cycle_idx', 'sample_nr']
        master = PD.read_csv('models/05_age/data.csv', sep = ';').loc[:, ['cycle_idx', 'is_lbw', 'piglet', 'age', 'age_log', 'session_idx', 'recording_idx']]
        master.set_index(['cycle_idx'], inplace = True)

        data = data.join(master, how = 'left', rsuffix = '_actual')

        data['age_diff'] = data['age'].values - data['age_actual'].values

        print ('storing individual prediciton diffs...')
        with open(storage_file, 'wb') as storage:
            PCK.dump(data, storage)
        print ('done storing! data is ready.')

    # print (data)
    return data

def UpdateLBWPredictions():
    data = GetPredictionData()
    lbw_data = data.loc[data['is_lbw'].values, :]
    # lbw_data = data.loc[NP.logical_not(data['is_lbw'].values), :]
    lbw_data.reset_index(inplace = True, drop = False)
    # print (lbw_data)
    # print (lbw_data.index.remove_unused_levels().levels[0].values)
    lbw_data.to_csv('results/lbw_strides.csv', sep = ';')
    strides = lbw_data.groupby(['piglet', 'age_actual', 'session_idx', 'recording_idx', 'cycle_idx'])
    # print (strides)
    strides = strides.agg({'age_diff': [NP.mean, NP.std, lambda ad: NP.sum(ad < 0.)/len(ad)]})#lambda arr: NP.sum(arr < 0)/len(arr)})
    print (strides)

    ax = SB.stripplot(  \
                    x = "age_diff" \
                  , y = "piglet" \
                  # , hue = "species" \
                  , data = lbw_data \
                  , dodge = True \
                  , alpha = .25 \
                  , zorder = 1 \
                  )
    ax.set_xlabel('age: prediction - actual difference')
    ax.set_ylabel('piglet')
    PLT.gcf().tight_layout()
    PLT.gcf().savefig('figures/lbw_predictions.png', dpi = 300, transparent = False)
    # PLT.show()
    PLT.close()

def PredictionComparisonTable():
    data = GetPredictionData()
    strides = data.groupby(['is_lbw', 'piglet', 'age_actual', 'cycle_idx']) # , 'session_idx', 'recording_idx'
    strides = strides.agg({'age_diff': [NP.mean, NP.std, lambda ad: NP.sum(ad < 0.)/len(ad)]})#lambda arr: NP.sum(arr < 0)/len(arr)})
    strides.columns = ['_'.join(col) for col in strides.columns]
    strides.reset_index(inplace = True, drop = False)
    strides.set_index(['cycle_idx'], inplace = True)
    # print (strides)

    is_lbw = strides['is_lbw'].values

    # (1) nbw strides
    nbw = strides.loc[NP.logical_not(is_lbw), :]
    nbw = nbw.groupby(['is_lbw']).agg({ \
                        'age_actual': lambda age: f'<{NP.mean(age):.1f}>' \
                      , 'piglet': len \
                      , 'age_diff_<lambda_0>': [lambda ad: NP.sum(ad>0.5), NP.mean ] \
                      , 'age_diff_mean': NP.mean \
                      , 'age_diff_std': NP.mean \
                      })
    nbw.index = ['<all NBW>']
    nbw.reset_index(inplace = True)
    columns = ['piglet' \
                   , 'age/h' \
                   , 'strides/' \
                   , 'underestimation/count' \
                   , 'underestimation/ratio' \
                   , 'pred. mean \(\Delta\)/h' \
                   , 'pred. std/h' \
                   ]
    nbw.columns = columns
    # print (nbw)

    # (2) lbw strides
    lbw = strides.loc[is_lbw, :]
    lbw.reset_index(inplace = True)
    # print (lbw)
    lbw = lbw.groupby(['piglet']).agg({ \
                        'age_actual': lambda age: f'{NP.mean(age):.1f}' \
                      , 'cycle_idx': len \
                      , 'age_diff_<lambda_0>': [lambda ad: NP.sum(ad>0.5), NP.mean ] \
                      , 'age_diff_mean': NP.mean \
                      , 'age_diff_std': NP.mean \
                      })

    print (lbw.columns)
    lbw.sort_values(('age_actual', '<lambda>'), inplace = True)

    lbw.reset_index(inplace = True)
    lbw.columns = columns
    # print (lbw)

    predictions = PD.concat([lbw, nbw], axis = 0) \
                    .set_index('piglet', inplace = False)

    predictions.index = [str(pig).replace('_', '.') for pig in predictions.index.values]
    predictions.columns = PD.MultiIndex.from_tuples([col.split('|') for col in predictions.columns])
    print (predictions)

    output = predictions.to_markdown(floatfmt = '.2f', tablefmt = "orgtbl", headers = columns)
    output = FormatTable(output)
    print (output)

    with open(f'results/prediction_lbw.org', 'w') as fi:
        print(output, file = fi)


if __name__ == "__main__":
    UpdateLBWPredictions()
    PredictionComparisonTable()
