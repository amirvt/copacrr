import itertools
import os

fold_names = ['fold01', 'fold02', 'fold03', 'fold04', 'fold05']

query_range = list(range(430, 449))# + list(range(601,700))

number_of_queries = len(query_range)

fold_len = number_of_queries // len(fold_names)

qid_year = dict(zip(
    query_range,
    list(itertools.chain(*[[fold] * fold_len for fold in fold_names]))
))



year_qids = {
    fold: query_range[(i * fold_len):((i+1)*fold_len)] for i, fold in enumerate(fold_names)
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
