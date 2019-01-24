import itertools
import os

fold_names = ['fold01', 'fold02', 'fold03', 'fold04', 'fold05']

query_start = 430
query_end = 449
number_of_queries = query_end - query_start + 1
fold_len = number_of_queries // len(fold_names)

qid_year = dict(zip(
    list(range(query_start, query_end + 1)),
    list(itertools.chain(*[[fold] * fold_len for fold in fold_names]))
))



year_qids = {
    fold: list(range(i * fold_len + query_start, i * fold_len + fold_len + query_start)) for i, fold in enumerate(fold_names)
}


# TODO what does this do?
def get_train_qids(fold, years=fold_names[:-1]):
    if fold.startswith('wt'):
        prefix = 'wt'
    elif fold.startswith('fold'):
        prefix = 'fold'
    a_qids = list()
    for y in fold[len(prefix):].split('_'):
        a_qids.extend(year_qids['%s%s' % (prefix, y)])
    return a_qids

# TODO what does this do?
def get_qrelf(basepath, year):
    # if year.startswith("nwt") or year.startswith("wt"):
    #     qrelf = os.path.join(basepath, 'qrels.adhoc.6y')
    # else:
    #     print("WARNING: no qrelf exists for get_train_qids on year: %s" % year)
    #     qrelf = None

    qrelf = os.path.join(basepath, 'qrels.adhoc.6y')

    return qrelf
