import sys
import random

def convert_relevance2label(relevance_str):
    relevance = int(relevance_str)
    return relevance
    if relevance < 2:
        return 0
    if relevance >2:
        return 1
    return 1 if random.randint(0,1) == 1 else 0
    

def extract_feature_and_label(inputfile):
    labels = []
    features = []
    with open(inputfile,'r') as inputstream:
        for line in inputstream:
            eles = line.strip().split()
            relevance_str = eles[0]
            label = convert_relevance2label(relevance_str)
            feat_strs = eles[2:]
            feat_vals = [float(feat_str.split(':')[1]) for feat_str in feat_strs]
            labels.append(label)
            features.append(feat_vals)
    return features,labels
            

if __name__ == "__main__":
    testfile = sys.argv[1]
    features,labels =  extract_feature_and_label(testfile)
    print sum(labels),len(labels)
