
import torch
import json
import pickle
import numpy as np
import os
import codecs
import random
from torch import nn
from torch.distributions import Categorical
from models import *
import utils
from utils import Indexer



################
# This Solver is used for pretraining word encoder

class EndingsSolver:
    
    def __init__(self, typ="ae", cmu_dict=None, args=None):
        self.last2_to_words = None
        self.typ = typ
        self.cmu_dict = cmu_dict
        self.args = args

    def init_model(self, y_start):
        
        #o_size=self.p_indexer.w_cnt, 
        model= Model(H=128, 
            i_size=self.g_indexer.w_cnt,
            o_size=self.g_indexer.w_cnt, 
            start_idx=y_start, 
            typ=self.args.model_type, 
            args=self.args)
        if self.args.use_cuda:
            model = model.cuda()
        # batch_size = args.batch_size # 64
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def get_splits(self, data_type='cmu_dict_words'):
        # self.all_cmu_items = all_items = list(self.cmu_data.items())
        if data_type == 'cmu_dict_words':
            all_items = list(self.cmu_dict.items())
            for i in range(11):
                random.shuffle(all_items)
            sz = len(all_items)
            train = all_items[:int(0.8*sz)]
            val = all_items[int(0.8*sz):int(0.9*sz)]
            test = all_items[int(0.9*sz):]
            self.splits = {'train':train,'val':val, 'test':test}
        elif data_type == 'sonnet':
            self.get_sonnet_splits(data_path=self.args.sonnet_data_path)
            self.splits = {}
            for k,vals in self.sonnet_splits.items():
                self.splits[k] = []
                for val in vals:
                    for w in val['ending_words']:
                        self.splits[k].append([w,None])
        elif data_type == 'limerick':
            self.get_limerick_splits(data_path=self.args.limerick_data_path)
            self.splits = {}
            for k, vals in self.limerick_splits.items():
                self.splits[k] = []
                for val in vals:
                    for w in val['ending_words']:
                        self.splits[k].append([w, None])
        else:
            assert False
        for k,v in self.splits.items():
            print("split ",k," has ", len(v), " items")
            print("split ",k," items[0:5] ", v[:5])


    def _load_sonnet_data(self, sonnet_data_path, split, skip_not_in_cmu=False):
        sonnet_data_file = sonnet_data_path + split + '.txt'
        data = open(sonnet_data_file,'r').readlines()
        ret = []
        skipped = 0
        for sonnet in data:
            lines = sonnet.strip().split(' <eos>')[:-1]
            last_words = [line.strip().split()[-1] for line in lines]
            if not skip_not_in_cmu:
                ret.append({'lines':lines, 'ending_words':last_words})
            else:
                skip=False
                for j,w in enumerate(last_words[:4]):
                    if w not in self.cmu_dict:
                        skipped+=1
                        skip=True
                        break
                if not skip:
                    ret.append({'lines':lines, 'ending_words':last_words})
        print("[_load_sonnet_data] : split=",split, " skipped ",skipped," out of ", len(data))
        return ret

    def _load_limerick_data(self, limerick_data_path, split):
        data_file = os.path.join(limerick_data_path, split + '_0.json')
        print("Loading from ", data_file)
        data = json.load(open(data_file, 'r'))
        ret = []
        skipped = 0
        n_tokens = 0
        for limerick in data:
            # if translator_limerick is not None:
            #     limerick = limerick['txt'].strip().translate(translator_limerick)
            # else:
            limerick = limerick['txt'].strip()
            lines = limerick.split('|')
            if len(lines) != 5:
                skipped += 1
                continue
            last_words = [line.strip().split()[-1] for line in lines]
            instance = {}
            instance['ending_words'] = last_words
            ret.append(instance)
        print("[_load_sonnet_data] : split=", split, " skipped ", skipped, " out of ", len(data))
        return ret

    def get_sonnet_splits(self, data_path="../data/sonnet_"):
        self.sonnet_splits = {k:self._load_sonnet_data(data_path,k) for k in ['train','valid','test'] }
        self.sonnet_splits['val'] = self.sonnet_splits['valid']
        # self.sonnet_splits = {k:self._process_sonnet_data(val) for k,val in self.sonnet_splits.items()}

    def get_limerick_splits(self, data_path="/data/limerick_"):
        self.limerick_splits = {k: self._load_limerick_data(data_path, k) for k in ['train', 'val', 'test']}
        # self.limerick_splits['val'] = self.limerick_splits['valid']
        # self.sonnet_splits = {k:self._process_sonnet_data(val) for k,val in self.sonnet_splits.items()}
        
    def index(self): 
        # why not use the vocab for indexing ?
        items = list(self.cmu_dict.items()) #self.all_cmu_items
        self.g_indexer = Indexer(self.args)
        self.g_indexer.process([i[0] for i in items])
        if self.typ!="ae":
            self.p_indexer = Indexer(self.args) 
            self.p_indexer.process([i[1] for i in items])  
                
    def save(self, dump_pre):
        pickle.dump(self.splits, open(dump_pre+'splits.pkl','wb'))
        pickle.dump(self.g_indexer, open(dump_pre+'g_indexer.pkl','wb'))
        self.g_indexer.save(dump_pre+'g_indexer')
        if self.typ!="ae":
            pickle.dump(self.p_indexer, open(dump_pre+'p_indexer.pkl','wb'))
            self.p_indexer.save(dump_pre+'p_indexer')

    def load(self, dump_pre):
        self.splits = pickle.load( open(dump_pre+'splits.pkl','rb'))
        if self.typ!="ae":
            self.p_indexer = Indexer(self.args) #pickle.load(open('tmp/tmp_'+args.g2p_model_name+'/solver_p_indexer.pkl','rb'))
            self.p_indexer.load(dump_pre+'p_indexer')
        self.g_indexer = Indexer(self.args) #pickle.load(open('tmp/tmp_'+args.g2p_model_name+'/solver_g_indexer.pkl','rb'))
        self.g_indexer.load(dump_pre+'g_indexer')
            
    def get_batch(self, i, batch_size, split, last_two=False, add_end_to_y=True): #typ='g2p',
        # typ: g2p, ae
        # assert typ==self.typ
        typ = self.typ
        if typ=="g2p":
            data = self.splits[split][i*batch_size:(i+1)*batch_size]
            x = [self.g_indexer.w_to_idx(g) for g,p in data]
            y = [self.p_indexer.w_to_idx(p) for g,p in data]
            y_start = self.p_indexer.w2idx[utils.START]
            y_end = self.p_indexer.w2idx[utils.END]
            if add_end_to_y:
                y = [y_i+[y_end] for y_i in y]
            x_end = self.g_indexer.w2idx[utils.END]                
            if self.args.use_eow_in_enc:
                x = [x_i+[x_end] for x_i in x]
            return x,y,y_start
        elif typ=="g2plast":
            data = self.splits[split][i*batch_size:(i+1)*batch_size]
            x = [self.g_indexer.w_to_idx(g) for g,p in data]
            #for g,p in data:
            #    print(p)
            y = [self.p_indexer.w_to_idx(p)[-2:] for g,p in data]
            y_start = self.p_indexer.w2idx[utils.START]
            y_end = self.p_indexer.w2idx[utils.END]
            if add_end_to_y:
                y = [y_i+[y_end] for y_i in y]
            if self.args.use_eow_in_enc:
                x_end = self.g_indexer.w2idx[utils.END]                
                x = [x_i+[x_end] for x_i in x]                
            return x,y,y_start
        elif typ=="ae":
            data = self.splits[split][i*batch_size:(i+1)*batch_size]
            x = [self.g_indexer.w_to_idx(g) for g,p in data]
            y = [self.g_indexer.w_to_idx(g) for g,p in data]
            y_start = self.g_indexer.w2idx[utils.START]
            y_end = self.g_indexer.w2idx[utils.END]
            if add_end_to_y:
                y = [y_i+[y_end] for y_i in y]
            if self.args.use_eow_in_enc:
                x_end = self.g_indexer.w2idx[utils.END]                
                x = [x_i+[x_end] for x_i in x]                
            return x,y,y_start
        
    def get_num_batches(self, split, batch_size):
        return int( ( len(self.splits[split]) + batch_size - 1.0)/batch_size )


    ################

    def get_loss(self, split, batch, batch_size, mode='train', use_alignment=False):

        model = self.model
        use_cuda = self.args.use_cuda
        batch_size = self.args.batch_size
        if mode=="train":
            model.train()
        else:
            model.eval()
        len_output = 0
        x,y,y_start = self.get_batch(i=batch, batch_size=batch_size, split=split) #, typ=args.data_type)
        
        batch_loss = torch.tensor(0.0)
        batch_align_loss = torch.tensor(0.0)
        batch_kl_loss = torch.tensor(0.0)
        if use_cuda:
            batch_loss = batch_loss.cuda()
            batch_align_loss = batch_align_loss.cuda()
            batch_kl_loss = batch_kl_loss.cuda()

        i=0
        all_e = []
        #print(" -- batch=",batch, " || x: ",len(x))
        for x_i,y_i in zip(x,y):
            
            len_output += len(y_i)
            e_i = model.encode(x_i)
            #print(i, e_i.size())
            all_e.append(e_i)
            out_all_i, info = model.decode(e_i,y_i)
            i+=1
            out_all_i = torch.stack(out_all_i)
            #dist = out_all_i.view(-1, self.p_indexer.w_cnt)
            dist = out_all_i.view(-1, self.g_indexer.w_cnt)
            targets = np.array(y_i, dtype=np.long)
            targets = torch.from_numpy(targets)
            if self.args.use_cuda:
                targets = targets.cuda()
            cur_loss = self.criterion(dist, targets)
            
            if use_alignment:
                y_i_all = self.p_indexer.idx_to_2(y_i)
                y_i_last2 = y_i_all[-3:-1] #solver.p_indexer.idx_to_2(y_i[-3:-1]) # remove end token ans use last2
                #x_j_word, success = solver.find_word_with_same_phoneme(y_i_last2)
                x_j_word, success = self.find_word_with_same_phoneme(y_i_all)
                #print(x_j_word, y_i_last2)
                if success:
                    x_j = self.g_indexer.w_to_idx(x_j_word)
                    e_j = model.encode(x_j)
                    cur_align_loss = torch.mean((e_i-e_j)*(e_i-e_j))
                    batch_align_loss += cur_align_loss
                    
            if model.typ=="vaed":
                batch_kl_loss+= info['kl_loss']
            
            #print(cur_loss)
            batch_loss += cur_loss
            
        total_loss = batch_align_loss + batch_loss + batch_kl_loss
        return total_loss, len_output, {'batch_align_loss':batch_align_loss.data.cpu().item(), \
                                        'total_batch_loss':total_loss.data.cpu().item(),\
                                        'batch_recon_loss':batch_loss.data.cpu().item(),\
                                       'batch_kl_loss': batch_kl_loss.cpu().item(),\
                                        'elbo_loss':(batch_loss.data.cpu().item()+batch_kl_loss.cpu().item())
                                       }
        

    def train(self, epochs=11, debug=False, use_alignment=False, args=None, model_dir=None):
        learning_rate = 1e-4
        model = self.model
        batch_size = self.args.batch_size
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = 999999999999.0
        
        for epoch in range(epochs):

            print("="*20, "[Training] beginning of epoch = ", epoch)

            all_loss_tracker = {}
            all_loss_tracker_val = {}
            
            num_batches = self.get_num_batches('train', batch_size)
            epoch_loss = 0.0
            ctr = 0
            for batch in range(num_batches):
                total_batch_loss, len_output, info = self.get_loss('train', batch, batch_size, mode='train',
                                                                   use_alignment=use_alignment)
                if debug:
                    print("TRAIN info = ", info)
                    print()
                for k,v in info.items():
                    if k.count('loss')>0:
                        if k not in all_loss_tracker:
                            all_loss_tracker[k] = 0.0
                        all_loss_tracker[k] += v
                epoch_loss = epoch_loss + total_batch_loss.data.cpu().item()
                model.zero_grad()
                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()
                ctr = ctr + len_output
                if batch%1000==0:
                    print("[Training] batch = ", batch, "epoch_loss = ", epoch_loss/ctr)
                if debug:
                    break
            print()
            print("[Training] epoch perplexity = ", np.exp(all_loss_tracker['elbo_loss']/ctr) )
            print("[Training] epoch all_loss_tracker (norm. by num_batches) = ",
                  {k:v/num_batches for k,v in all_loss_tracker.items()} )
            print()

            num_batches = self.get_num_batches('val', batch_size)
            epoch_val_loss = 0.0
            ctr = 0
            for batch in range(num_batches):
                total_batch_loss, len_output, info = self.get_loss('val', batch, batch_size, mode='eval', use_alignment=use_alignment)
                if debug:
                    print("VAL info = ", info)
                epoch_val_loss = epoch_val_loss + total_batch_loss.data.cpu().item()
                for k,v in info.items():
                    if k.count('loss')>0:
                        if k not in all_loss_tracker_val:
                            all_loss_tracker_val[k] = 0.0
                        all_loss_tracker_val[k] += v
                ctr = ctr + len_output
                if debug:
                    break
            print("[Training] epoch VAL epoch_val_loss  = ", epoch_val_loss)
            if 'elbo_loss' in all_loss_tracker_val:
                print("[Training] epoch VAL perplexity = ", np.exp(all_loss_tracker_val['elbo_loss']/ctr) )
            print("\n[Training] epoch all_loss_tracker_val (norm. by num_batches) = ", {k:v/num_batches for k,v in all_loss_tracker_val.items()} )
            if True: #not debug:
                print("\n[Training] Saving model at ", model_dir + 'model_' + str(epoch%5) )
                torch.save(model.state_dict(), model_dir + 'model_' + str(epoch%5))
                if (epoch_val_loss/ctr) < best_loss:
                    best_loss = epoch_val_loss/ctr
                    torch.save(model.state_dict(), model_dir + 'model_best')
                    print("\n[Training] Saving best model till now at ", model_dir + 'model_best')
            if debug:
                break
            print()
            
    ################

    def analyze(self, vocab):
        
        ##load
        model = self.model
        args = self.args
        batch_size = args.batch_size
        model_dir = 'tmp/tmp_'+args.model_name+'/'
        assert os.path.exists(model_dir)
        self.load(model_dir + 'solver_')
        state_dict_best = torch.load(model_dir+'model_best')
        model.load_state_dict(state_dict_best)
        self.load(model_dir+'solver_')

        ##utils
        def fnc(s1,s2):
            #x,y,y_start = [s1,s2],None,y_start
            s1 = self.g_indexer.w_to_idx(s1)
            s2 = self.g_indexer.w_to_idx(s2)
            model.eval()
            e1 = model.encode(s1)
            e2 = model.encode(s2)
            #y = [self.p_indexer.w_to_idx(p) for g,p in data]
            e1_numpy = e1.data.cpu().numpy().reshape(-1)
            e2_numpy = e2.data.cpu().numpy().reshape(-1)
            return np.sum(e1_numpy * e2_numpy)/ np.sqrt( (np.sum(e1_numpy * e1_numpy) * np.sum(e2_numpy * e2_numpy)) )
        
        def pred(s1):
            model.eval()
            e_1 = model.encode( self.g_indexer.w_to_idx(s1) )
            out_all_i,_ = model.decode(e_1,'',use_gt=False)
            print(out_all_i)
            return self.p_indexer.idx_to_2(out_all_i)

        ## simple analysis
        for word_pair in [['red','head'], ['glue','blue'],['red','red'],['apple','blue'],['table','able'],['tram','cram']]:
            if args.data_type!="ae":
                print("pred: word_pair[0] ", word_pair[0], " => ", pred(word_pair[0]), " || ", \
                    "pred: word_pair[1] ", word_pair[1], " => ", pred(word_pair[1]) )
            print("word_pair = ", word_pair, " --fnc(): ", fnc(word_pair[0],word_pair[1]))
            print()

        ## compute loss vals
        num_batches = self.get_num_batches('test', batch_size)
        epoch_val_loss = 0.0
        ctr = 0
        all_loss_tracker_val = {}
        for batch in range(num_batches):
            total_batch_loss, len_output, info = self.get_loss('test', batch, batch_size, mode='eval', use_alignment=args.use_alignment)
            epoch_val_loss = epoch_val_loss + total_batch_loss.data.cpu().item()
            for k,v in info.items():
                if k.count('loss')>0:
                    if k not in all_loss_tracker_val:
                        all_loss_tracker_val[k] = 0.0
                    all_loss_tracker_val[k] += v
            ctr = ctr + len_output
        print("TEST epoch_val_loss  = ", epoch_val_loss)
        if 'elbo_loss' in all_loss_tracker_val:
            print("TEST perplexity = ", np.exp(all_loss_tracker_val['elbo_loss']/ctr) )
        print("TEST all_loss_tracker_val (norm. by num_batches) = ", {k:v/num_batches for k,v in all_loss_tracker_val.items()} )

        #dump embs
        all_embs = {}
        dump_file = model_dir + "all_embs.pkl"
        print("dumping to ",dump_file)
        vocab[0:5], len(vocab)
        model.eval()
        for w in vocab:
            try:
                s1 = self.g_indexer.w_to_idx(w.lower())
            except:
                print("error for ",w)
                continue
            e1 = model.encode(s1)
            e1_numpy = e1.data.cpu().numpy().reshape(-1)
            all_embs[w] = e1_numpy
            #break
        pickle.dump(all_embs, open(dump_file,'wb'))

    ################



