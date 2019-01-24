# %%
import datetime
import getpass
import itertools
import os
import sys
import time
import subprocess

# all_years = ['09', '10', '11', '12', '13', '14']
# #train_test_years = {'wt12_13':['wt11', 'wt14']}
# train_test_years = {'wt' + '_'.join(sorted(years)):
#         sorted(['wt' + ty for ty in all_years if ty not in years])
#         for years in itertools.combinations(all_years, 4)}


# train_test_years = \
#     {
#         'wt09_10_11_12': ['wt13', 'wt14'],
#         'wt09_10_11_13': ['wt12', 'wt14'],
#         'wt09_10_11_14': ['wt12', 'wt13'],
#         'wt09_10_12_13': ['wt11', 'wt14'],
#         'wt09_10_12_14': ['wt11', 'wt13'],
#         'wt09_10_13_14': ['wt11', 'wt12'],
#         'wt09_11_12_13': ['wt10', 'wt14'],
#         'wt09_11_12_14': ['wt10', 'wt13'],
#         'wt09_11_13_14': ['wt10', 'wt12'],
#         'wt09_12_13_14': ['wt10', 'wt11'],
#         'wt10_11_12_13': ['wt09', 'wt14'],
#         'wt10_11_12_14': ['wt09', 'wt13'],
#         'wt10_11_13_14': ['wt09', 'wt12'],
#         'wt10_12_13_14': ['wt09', 'wt11'],
#         'wt11_12_13_14': ['wt09', 'wt10']
#     }


train_test_years = \
    {
        'fold01_02_03': ['fold04', 'fold05'],

        'fold01_02_04': ['fold03', 'fold05'],

        'fold01_02_05': ['fold03', 'fold04'],
        'fold01_03_04': ['fold02', 'fold05'],

        'fold01_03_05': ['fold02', 'fold04'],
        'fold01_04_05': ['fold02', 'fold03'],
        'fold02_03_04': ['fold01', 'fold05'],

        'fold02_03_05': ['fold1', 'fold04'],
        'fold02_04_05': ['fold1', 'fold03'],
        'fold03_04_05': ['fold1', 'fold02']
    }


os.environ['parentdir'] = '/home/amir/PycharmProjects/copacrr/out'

for train_years, (valid_year, test_year) in train_test_years.items():
    os.environ['train_years'] = train_years
    os.environ['test_year'] = test_year
    os.environ['valid_years'] = valid_year
    print('*******')
    print('trains_years=', train_years)
    print('*******')
    process = subprocess.Popen(['bash', './bin/train_model.sh'], env=dict(os.environ, train_years=train_years))
    process.wait()
    print('*******')
    # print('trains_years=', train_years)
    # print('test_year=', test_year)
    # print('*******')
    # process = subprocess.Popen(['bash', './bin/pred_per_epoch.sh'], env=dict(os.environ, train_years=train_years, test_year=test_year))
    # process.wait()
    # print('*******')
    # print('trains_years=', train_years)
    # print('test_year=', valid_year)
    # print('*******')
    # process = subprocess.Popen(['bash', './bin/pred_per_epoch.sh'], env=dict(os.environ, train_years=train_years, test_year=valid_year))
    # process.wait()
