def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    cnt = 0
    for i in relevant:
        if i in recommended[:k]:
           cnt += 1
    return [cnt/k,cnt/len(relevant)]