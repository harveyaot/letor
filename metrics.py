import numpy as np

# change this if using K > 100
denominator_table = np.log2( np.arange( 2, 102 ))
exception_gain = lambda relevance : 2 ** relevance - 1
exception_gain_vectorize = np.vectorize(exception_gain)

def calculate_DCG_at_k( r, k):
	"""Score is discounted cumulative gain (DCG)
	Relevance is positive real values.
        Refer to
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
	Args:
		r: Relevance scores (list or numpy) in rank order
			(first element is the first item)
		k: Number of results to consider

	Returns:
		Discounted cumulative gain
	"""
	r = np.asarray(r)[:k]
	if r.size:
            # return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return np.sum( exception_gain_vectorize(r) / 
                    denominator_table[:r.shape[0]] )
	return 0.
 
 
def calculate_NDCG_at_k ( r, k):
	"""Score is normalized discounted cumulative gain (NDCG)

	Relevance orignally was positive real values.
	Args:
		r: Relevance scores (list or numpy) in rank order
			(first element is the first item)
		k: Number of results to consider

	Returns:
		Normalized discounted cumulative gain
	"""
	DCG_max = calculate_DCG_at_k(sorted(r, reverse=True), k)
	DCG_min = calculate_DCG_at_k(sorted(r), k)
	assert( DCG_max >= DCG_min )
	
	if not DCG_max:
		return 0.
	 
	DCG = calculate_DCG_at_k(r, k)
	
	#print DCG_min, DCG, DCG_max
	return (DCG - DCG_min) / ((DCG_max - DCG_min) + 1)
	#return DCG / DCG_max
