Expriments on Microsoft's learning to rank dataset
====================



1. Metrics:(NCDG) implement in metrics.py

2. Experiments Road

    1. Using static method like(BM25, CTR, TFIDF)

    2. Try Point-Wise method on  Linear model like(Logistic Regression) and Trees like(GBDT) to predict every query-url score

    3. Try Pair-Wise method

    4. Try List-Wise method

    5. Try Deep Learning method in Ranking

3. Coding Explaination:

    1. `rank.py` contains all rankers

    2. `metrics.py` implements the NCDG metrics for meature performance.

    3. `ml_headquater.py` using scikit-learn to train and predict ml ranker model 

    4. `making.py` making feature vectors from [letor data](http://research.microsoft.com/en-us/projects/mslr/)

    5. `config.py` some basic config on logging moudle and CONST

    6. `validate_rank.py` compare the performance of rankers.

