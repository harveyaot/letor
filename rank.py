# author wei he
import numpy as np
import metrics
import lg_model

from itertools import groupby
from ml_headquater import load_model

DEFAULT_NDCG_EVAL_POSITION = 20

def extract_feature_val_from_str(feature_str):
    return float(feature_str.split(":")[1])

class Ranker(object):
    """The super class of all rankers.
    Responsible for load testing data. 

    """
    def __init__(self,dataset):
        """
            Args:
                dataset: file stream of testing dataset
        """
        self.ranked_qd_groups = []
        self.qd_groups = []
        self.dataset = dataset

    def dataset_generator(self):
        with open(self.dataset,'r') as dataset:
            for line in dataset:
                yield line.strip().split(' ')

    def get_next_qd_group(self):
        for queryid, qd_group in groupby(self.dataset_generator(), lambda x : x[1]):
            yield qd_group

    def _calulte_mean_NCDG_origin_qd_groups(self,qd_groups):
        ndcg_score_arr = np.empty(len(qd_groups))
        for i, qd_group in enumerate(qd_groups):
            relevance = np.asarray([int(qd_record[0]) for qd_record in 
                                                                    qd_group]) 
            ndcg_score_arr[i] = metrics.calculate_NDCG_at_k(relevance,
                                                        DEFAULT_NDCG_EVAL_POSITION)
        return  np.mean(ndcg_score_arr)

    def calculate_mean_NCDG_for_dataset(self):
        return self._calulte_mean_NCDG_origin_qd_groups(self.ranked_qd_groups)

    def calculate_mean_NCDG_for_original_dataset(self):
        return self._calulte_mean_NCDG_origin_qd_groups(self.qd_groups)

    def rank_dataset(self):
        self.ranked_qd_groups = []
        for qd_group in self.get_next_qd_group():
            ranked_qd_group = self.rank_qd_group(qd_group)
            self.ranked_qd_groups.append(ranked_qd_group)
            self.qd_groups.append(qd_group)

    def rank_qd_group(self,qd_group):
        ranked_qd_group = sorted(qd_group,key=lambda qd : self.relevance_of_qd(qd),
                                                            reverse = True)
        return ranked_qd_group


class TFIDF_Ranker(Ranker):
    def relevance_of_qd(self,qd_record):
        # features from 86 - 90 are about mean tfidf
        score =  sum(map(extract_feature_val_from_str,
                        [feature_str for feature_str in qd_record[87:92]]))
        return score

class BM25_Ranker(Ranker):
    def relevance_of_qd(self,qd_record):
        # feature_of_BM25 from 106 to 110
        return sum(map(extract_feature_val_from_str,
                        [feature_str for feature_str in qd_record[107:112]]))

class Static_Ranker(Ranker):
    def relevance_of_qd(self,qd_record):
        # feature_of_PAGERANK 130
        # feature_of_qu click_count
        pagerank,qd_click = map(extract_feature_val_from_str,[qd_record[131],qd_record[135]])
        return pagerank * qd_click

class ML_Ranker(Ranker):
    def __init__(self,dataset,model_file):
        super(ML_Ranker,self).__init__(dataset)
        self.model = load_model(model_file)

    def relevance_of_qd(self,qd_record):
        features = map(extract_feature_val_from_str,qd_record[2:])
        score = np.dot(self.model.classes_,self.model.predict_proba([features])[0])
        return score
