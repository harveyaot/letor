import sys

from rank import TFIDF_Ranker
from rank import BM25_Ranker
from rank import Static_Ranker
from rank import ML_Ranker 

ml_model = "gbdt_sample.md"
ml_model = "models/gbdt_70wtrain_1000tree_8depth.md"

def validate():
    ml_ranker = ML_Ranker(dataset_file,ml_model) 
    #tfidf_ranker = TFIDF_Ranker(dataset_file) 
    bm25_ranker = BM25_Ranker(dataset_file) 
    #static_ranker = Static_Ranker(dataset_file)

    ml_ranker.rank_dataset()
    #tfidf_ranker.rank_dataset()
    bm25_ranker.rank_dataset()
    #static_ranker.rank_dataset()

    print  "ml_ranker: %.2f"%ml_ranker.calculate_mean_NCDG_for_dataset()
    
    #print  "tfidf: %.2f"%tfidf_ranker.calculate_mean_NCDG_for_dataset()
    print  "bm25:  %.2f"%bm25_ranker.calculate_mean_NCDG_for_dataset()
    print  "random_ranker: %.2f"%ml_ranker.calculate_mean_NCDG_for_original_dataset()
    #print  "static:%.2f"%static_ranker.calculate_mean_NCDG_for_dataset()


def usage():
    print >> sys.stderr,"python ", sys.argv[0],"dataset_file"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)

    dataset_file = sys.argv[1]
    validate()
