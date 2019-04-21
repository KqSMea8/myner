
import torch
import torch.utils.data
import codecs
import tqdm
import random
target_data = './target.txt'
source_path = './source.txt'
target_path = './target.txt'
seq_data = './source.txt'

class NerDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, target_path, device, train_ratio=0.85):
        super(NerDataset, self).__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.length = 0
        self.dataset = None
        self.trainset = None
        self.testset = None
        self.train_ratio = train_ratio
        self.device = device
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        with codecs.open(self.source_path, 'r', encoding='utf-8') as s:
            with codecs.open(self.target_path, 'r', encoding='utf-8') as f:
                s_list = [_ for _ in s]
                f_list = [_ for _ in f]

        self.dataset = list(zip(s_list, f_list))
        random.shuffle(self.dataset)
        self.length = len(self.dataset)
        self.train_size = int(self.length*self.train_ratio)
        self.trainset = self.dataset[:self.train_size-1]
        self.testset = self.dataset[self.train_size:]
        self.word2id, self.target2id = self.create_map_id()
        self.id2target = {v:k for k, v in self.target2id.items()}

        self.target2id[self.START_TAG] = len(self.target2id)
        self.target2id[self.STOP_TAG] = len(self.target2id)
        self.type = 'train'

    def __len__(self):
        if self.type == "train":
            return self.train_size-1
        else:
            return self.length - self.train_size+1
    def __getitem__(self, index):
        s, t = None, None
        if self.type == 'train':
            s, t = self.trainset[index]
        elif  self.type == 'test':
            s, t = self.testset[index]




        s = self.prepare_tensor_data(s.split(), 'seq')
        t = self.prepare_tensor_data(t.split(), 'tag')
        return [s, t]


    def set_mode(self, type):
        self.type = type

    def create_map_id(self):
        word2id = {}
        target2id ={}
        for data in self.dataset:
            word_arr = data[0].split()
            for word in word_arr:
                if word not in word2id:
                    word2id[word] = len(word2id)
            target_arr = data[1].split()
            for word in target_arr:
                if word not in target2id:
                    target2id[word] = len(target2id)
        return word2id, target2id



    def prepare_tensor_data(self, seq, type):
        if type == "seq":
            idxs = [self.word2id[w] for w in seq]
            # return torch.Tensor(idxs).long().to(self.device)
            return torch.Tensor(idxs).long()
        elif type == "tag":
            idxs = [self.target2id[w] for w in seq]
            # return torch.Tensor(idxs).long().to(self.device)
            return torch.Tensor(idxs).long()

    def tagid2ner(self, id_list):
        return [self.id2target[id] for id in id_list]


    def get_word2id(self):
        return self.word2id

    def get_target2id(self):
        return self.target2id


    def get_len(self):
        return self.__len__()
    # def convert2tensor(self, batch_data):



def main():
    nerdataset = NerDataset(source_path, target_path, "cpu")
    print(nerdataset.__getitem__(3))

    mydata_loader = torch.utils.data.DataLoader(nerdataset, batch_size=1, shuffle=False, num_workers=1)
    for batch in tqdm.tqdm(mydata_loader):
        #s_batch, t_batch = nerdataset.convert2tensor(batch)
        s, t = batch
        s = s.squeeze(0)
        t = t.squeeze(0)
        print(s.size())
        break
    pass

# if __name__ == '__main__':
#     main()