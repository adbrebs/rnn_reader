import cPickle
import os
from itertools import cycle
import matplotlib.pyplot as plt


parent_path = '/Users/adeb/results/QA/'

FOLDERS = [
    ('993235127187', 'GRU concat'),
    ('756555092286', 'attention cheap'),
    ('356855230263', 'attention softmax'),

    # ('res_2_no_attention_343181', 'GRU concat 2 res'),
    # ('res_2_attention_softmax_667446', 'attention softmax 2 res'),
    # ('res_2_attention_efficient_small_267959', 'attention efficient 2 res'),

    # ('skip_2_attention_efficient_small_714087', 'attention efficient 2 skip'),
    # ('skip_2_attention_softmax_623583', 'attention softmax 2 skip'),
    # ('skip_2_no_attention_614797', 'GRU concat 2 skip'),

    ('bdir_attention_efficient_small_1014', 'bidir efficient'),
    ('bdir_attention_softmax_276805', 'bidir softmax'),
    ('bdir_no_attention_132591', 'bidir GRU concat'),
]

xlim = (0, 140000)

FEATURE_NAME = 'accuracy'
SETS = ['valid']


fig = plt.figure(num=None, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
cmap = plt.get_cmap('prism')
# colors = [cmap(i) for i in np.linspace(0, 1, 10*len(FOLDERS))]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
colorcyler = cycle(colors)

for (folder, name) in FOLDERS:
    c = next(colorcyler)
    t = next(linecycler)
    for i, set in enumerate(SETS):
        f = os.path.join(parent_path, folder, 'var_saver_' + set + '.pkl')
        data = cPickle.load(open(f, 'r'))

        names = data['names']
        if FEATURE_NAME in names:
            id_feature = names.index(FEATURE_NAME)
        else:
            continue
        if not data['history'].any():
            continue
        feature = data['history'][:, id_feature]

        plt.plot(data['iterations'], feature,
                 '.' + t, color=c, label=set + '_' + name)

plt.xlabel('Number of minibatches')
plt.ylabel(FEATURE_NAME)
plt.legend(loc='best')
plt.title(FEATURE_NAME)
plt.xlim(xlim)
plt.show()
