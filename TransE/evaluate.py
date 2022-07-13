import torch

# hits@k
def hit_at_k(preds, gt_idx, device, k):
    if preds.size(0) == gt_idx.size(0):
        zero_tensor = torch.tensor([0],device = device)
        one_tensor = torch.tensor([1], device = device)
        _, idx = preds.topk(k=k, largest = False) #returns values, indices
    else:
        raise AssertionError
    return torch.where(idx == gt_idx, one_tensor,zero_tensor).sum().item()


# mrr: mean reciprocal rank
def mrr(preds, gt_idx):
    '''
    :param preds: B * N tensor of prediction values B: batch size, N: number of classes, sorted in class idx order
    :param gt_idx: B * 1 tensor with index of gt class
    :return: mrr score
    '''
    if preds.size(0) == gt_idx.size(0):
        tmp = gt_idx.reshape(-1, 1)
        gt_idx = tmp.expand(preds.size())
        correct = (gt_idx == preds).nonzero()
        ranks = correct[:,-1] + 1
        ranks = ranks.float()
        r_ranks = torch.reciprocal(ranks)
        mrr = torch.sum(r_ranks).data / gt_idx.size(0)
    return mrr.item()

