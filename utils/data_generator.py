import numpy as np
import keras
from collections import Counter
import logging

logger = logging.getLogger('pacrr')


class MY_Generator(keras.utils.Sequence):

    def __init__(self, select_pos_func, dim_sim, max_query_term, n_grams, doc_mat_dir, qid_wlen_cwid_mat,
                 qid_cwid_label, \
                 query_idfs, sample_qids, binarysimm, \
                 label2tlabel={4: 2, 3: 2, 2: 2, 1: 1, 0: 0, -2: 0}, \
                 sample_label_prob={2: 0.5, 1: 0.5}, \
                 n_query_terms=16, \
                 NUM_NEG=10, \
                 n_dims=300, n_batch=32, random_shuffle=True, random_seed=14, qid_context=None,
                 feature_names=["sims"]):
        self.n_batch = n_batch
        self.sample_label_prob = sample_label_prob
        self.sample_qids = sample_qids
        self.qid_wlen_cwid_mat = qid_wlen_cwid_mat
        self.qid_context = qid_context
        self.NUM_NEG = NUM_NEG
        self.query_idfs = query_idfs
        self.n_query_terms = n_query_terms
        self.random_shuffle = random_shuffle
        self.binarysimm = binarysimm
        self.doc_mat_dir = doc_mat_dir
        self.feature_names = feature_names
        self.dim_sim = dim_sim
        self.max_query_term = max_query_term
        self.n_grams = n_grams
        self.select_pos_func = select_pos_func

        np.random.seed(random_seed)
        self.context = qid_context is not None
        self.qid_label_cwids = dict()
        label_count = dict()
        label_qid_count = dict()
        for qid in sample_qids:
            if qid not in qid_cwid_label or qid not in self.qid_wlen_cwid_mat:
                logger.error('%d in qid_cwid_label %r, in qid_cwid_mat %r' % \
                             (qid, qid in qid_cwid_label, qid in self.qid_wlen_cwid_mat))
                continue
            self.qid_label_cwids[qid] = dict()
            wlen_k = list(self.qid_wlen_cwid_mat[qid].keys())[0]
            print(qid_cwid_label)
            for cwid in qid_cwid_label[qid]:
                l = label2tlabel[qid_cwid_label[qid][cwid]]
                if cwid not in self.qid_wlen_cwid_mat[qid][wlen_k]:
                    print(cwid, self.qid_wlen_cwid_mat[qid][wlen_k])
                    logger.error('%s not in %d in self.qid_wlen_cwid_mat' % (cwid, qid))
                    continue
                if l not in self.qid_label_cwids[qid]:
                    self.qid_label_cwids[qid][l] = list()
                self.qid_label_cwids[qid][l].append(cwid)
                if l not in label_qid_count:
                    label_qid_count[l] = dict()
                if qid not in label_qid_count[l]:
                    label_qid_count[l][qid] = 0
                label_qid_count[l][qid] += 1
                if l not in label_count:
                    label_count[l] = 0
                label_count[l] += 1

        if len(sample_label_prob) == 0:
            total_count = sum([label_count[l] for l in label_count if l > 0])
            sample_label_prob = {l: label_count[l] / float(total_count) for l in label_count if l > 0}
            logger.error('nature sample_label_prob', sample_label_prob)
        label_qid_prob = dict()
        for l in label_qid_count:
            if l > 0:
                total_count = label_count[l]
                label_qid_prob[l] = {qid: label_qid_count[l][qid] / float(total_count) for qid in
                                     label_qid_count[l]}
        self.sample_label_qid_prob = {
            l: [label_qid_prob[l][qid] if qid in label_qid_prob[l] else 0 for qid in sample_qids] \
            for l in label_qid_prob}

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def get_doc_matrix(qid, cwid, doc_mat_dir, feature_names, n_grams, dim_sim, max_query_term, select_pos_func):
        topic_cwid_fs = [doc_mat_dir + '/topic_doc_mat/%s/%d/%s.npy' % (fname, qid, cwid) for fname in feature_names]

        topic_mat = [np.load(topic_cwid_f).astype(np.float32) for topic_cwid_f in topic_cwid_fs]
        for i in range(len(topic_mat)):
            if len(topic_mat[i].shape) == 1:
                topic_mat[i] = np.expand_dims(topic_mat[i], axis=0)[:, :-1]
            else:
                topic_mat[i] = topic_mat[i][:, :-1]
            topic_mat[i] = np.nan_to_num(topic_mat[i], 0)

        empty = False
        shape = topic_mat[0].shape
        for mat in topic_mat:
            if mat.shape != shape or mat.shape[1] == 0:
                empty = True

        if not empty:
            raw_res = np.array([mat for mat in topic_mat]).astype(np.float32)
            raw_res = np.moveaxis(raw_res, 0, -1)

        else:
            logger.error('dimension mismatch {0} {1} {2}'.format(qid, cwid, topic_mat[0].shape))

        res = dict()
        pad_value = 0
        for n_gram in n_grams:
            len_doc = raw_res.shape[1]
            len_query = raw_res.shape[0]

            if len_doc > dim_sim:
                rmat = np.pad(raw_res, pad_width=((0, max_query_term - len_query), (0, 1), (0, 0)),
                              mode='constant', constant_values=pad_value).astype(np.float32)
                selected_inds = select_pos_func(res, dim_sim, n_gram)
                res[n_gram] = (rmat[:, selected_inds, :])
            else:
                res[n_gram] = (np.pad(raw_res, pad_width=((0, max_query_term - len_query),
                                                          (0, dim_sim - len_doc), (0, 0)), mode='constant',
                                      constant_values=pad_value).astype(np.float32))

        return raw_res

    def __getitem__(self, idx):

        pos_batch = dict()
        neg_batch = dict()
        qid_batch = list()
        pcwid_batch = list()
        ncwid_batch = list()
        qidf_batch = list()
        pos_context_batch = []
        neg_context_batch = {}
        ys = list()
        selected_labels = np.random.choice([l for l in sorted(self.sample_label_prob)],
                                           size=self.n_batch, replace=True,
                                           p=[self.sample_label_prob[l] for l in sorted(self.sample_label_prob)])
        label_counter = Counter(selected_labels)
        total_train_num = 0
        for label in label_counter:
            nl_selected = label_counter[label]
            if nl_selected == 0:
                continue
            selected_qids = np.random.choice(self.sample_qids,
                                             size=nl_selected, replace=True, p=self.sample_label_qid_prob[label])
            qid_counter = Counter(selected_qids)
            for qid in qid_counter:
                pos_label = 0
                nq_selected = qid_counter[qid]
                if nq_selected == 0:
                    continue
                for nl in reversed(range(label)):
                    if nl in self.qid_label_cwids[qid]:
                        pos_label = label
                        neg_label = nl
                        break
                if pos_label != label:
                    continue
                pos_cwids = self.qid_label_cwids[qid][label]
                neg_cwids = self.qid_label_cwids[qid][nl]
                n_pos, n_neg = len(pos_cwids), len(neg_cwids)
                idx_poses = np.random.choice(list(range(n_pos)), size=nq_selected, replace=True)
                min_wlen = min(self.qid_wlen_cwid_mat[qid])

                for pi in idx_poses:
                    p_cwid = pos_cwids[pi]
                    pos_mats = self.get_doc_matrix(qid, p_cwid, self.doc_mat_dir, self.feature_names, self.n_grams,
                                              self.dim_sim,
                                              self.max_query_term, self.select_pos_func)
                    for wlen in self.qid_wlen_cwid_mat[qid]:
                        if wlen not in pos_batch:
                            pos_batch[wlen] = list()

                        pos_batch[wlen].append(pos_mats[wlen])
                        if wlen == min_wlen:
                            if self.qid_wlen_cwid_mat:
                                pos_context_batch.append(self.qid_context[qid][p_cwid])
                            ys.append(1)
                for neg_ind in range(self.NUM_NEG):
                    idx_negs = np.random.choice(list(range(n_neg)), size=nq_selected, replace=True)
                    min_wlen = min(self.qid_wlen_cwid_mat[qid])
                    for ni in idx_negs:
                        n_cwid = neg_cwids[ni]
                        neg_mats = self.get_doc_matrix(qid, n_cwid, self.doc_mat_dir, self.feature_names, self.n_grams,
                                                  self.dim_sim,
                                                  self.max_query_term, self.select_pos_func)
                        for wlen in self.qid_wlen_cwid_mat[qid]:
                            if wlen not in neg_batch:
                                neg_batch[wlen] = dict()
                            if neg_ind not in neg_batch[wlen]:
                                neg_batch[wlen][neg_ind] = list()
                                if wlen == min_wlen and self.qid_wlen_cwid_mat:
                                    neg_context_batch[neg_ind] = []

                            neg_batch[wlen][neg_ind].append(neg_mats[wlen])
                            if wlen == min_wlen and self.qid_wlen_cwid_mat:
                                neg_context_batch[neg_ind].append(self.qid_context[qid][n_cwid])
                qidf_batch.append(self.query_idfs[qid].reshape((1, self.query_idfs, 1)).repeat(nq_selected, axis=0))
        total_train_num = len(ys)
        if self.random_shuffle:
            shuffled_index = np.random.permutation(list(range(total_train_num)))
        else:
            shuffled_index = list(range(total_train_num))
        train_data = dict()
        labels = np.array(ys)[shuffled_index]

        getmat = lambda x: np.array(x)

        for wlen in pos_batch:
            train_data['pos_wlen_%d' % wlen] = getmat(pos_batch[wlen])[shuffled_index, :]
            for neg_ind in range(self.NUM_NEG):
                train_data['neg%d_wlen_%d' % (neg_ind, wlen)] = np.array(getmat(neg_batch[wlen][neg_ind]))[
                                                                shuffled_index, :]

        if self.binarysimm:
            for k in train_data:
                assert k.find("_wlen_") != -1, "data contains non-simmat objects"
                train_data[k] = (train_data[k] >= 0.999).astype(np.int8)

        if self.qid_wlen_cwid_mat:
            train_data['pos_context'] = np.array(pos_context_batch)[shuffled_index]
            for neg_ind in range(self.NUM_NEG):
                train_data['neg%d_context' % neg_ind] = np.array(neg_context_batch[neg_ind])[shuffled_index]

        train_data['query_idf'] = np.concatenate(qidf_batch, axis=0)[shuffled_index, :]

        train_data['permute'] = np.array([[(bi, qi) for qi in np.random.permutation(self.query_idfs)]
                                          for bi in range(self.n_batch)], dtype=np.int)
        return (train_data, labels)
