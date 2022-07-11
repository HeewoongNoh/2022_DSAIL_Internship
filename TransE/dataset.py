#import library and package
from collections import Counter
from torch.utils import data

# Create separate mappings to indices for entities and relations
def create_mapping(path):
    e_counter = Counter()
    r_counter = Counter()
    e_to_id = {}
    r_to_id = {}
    with open(path,"r") as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            e_counter.update([head,tail])
            r_counter.update([relation])
    for idx, (mid,_) in enumerate(e_counter.most_common()):
        e_to_id[mid] = idx
    for idx, (relation,_) in enumerate(r_counter.most_common()):
        r_to_id[relation] = idx
    return e_to_id, r_to_id

#Dataset class for FB15K-237
class FB15K(data.Dataset):
    def __init__(self,path,e_to_id,r_to_id):
        self.e_to_id = e_to_id
        self.r_to_id = r_to_id
        self.data = []
        with open(path,"r") as f:
            #triplets (head, relation, tail)
            for line in f:
                self.data.append(line[:-1].split("\t"))

    # return number of dataset
    def __len__(self):
        return len(self.data)

    # return (head id, relation id, tail id)
    def __getitem__(self,idx):
        head, relation, tail = self.data[idx]
        head_id = self.e_to_id[head]
        relation_id = self.r_to_id[relation]
        tail_id = self.e_to_id[tail]
        return head_id, relation_id, tail_id

# print(create_mapping('FB15K-237/train.txt'))

