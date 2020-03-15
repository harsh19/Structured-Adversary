import time
import string
import torch
import json
import pickle
import numpy as np
import ipdb as pdb
import os
import codecs
import random
from torch import nn
from torch.distributions import Categorical
from models_lm import *
from models import *
import utils
from utils import *
from utils_lm import *
from allennlp.common.tqdm import Tqdm
import signal
from contextlib import contextmanager
from typing import Tuple
from constants import *
from rhyming_eval import RhymingEval


translator = None
translator_limerick = None
# translator = str.maketrans('', '', string.punctuation.replace('<','').replace('>',''))
# translator_limerick = str.maketrans('', '', string.punctuation.replace('|',''))
_couplet_mode = True     # Setting it to True uses vanilla-lm for couplets

def load_sonnet_vocab(pth):
    data = [r.strip() for r in open(pth,"r").readlines()]
    return data

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



################

class MainSolver:

    def __init__(self, 
            typ="g2p", 
            cmu_dict=None, 
            args=None, 
            vocab_type="only_ending_words",
            mode='train'):

        self.last2_to_words = None
        self.typ = typ
        self.cmu_dict = cmu_dict
        self.args = args
        self.sonnet_splits = None
        self.limerick_splits = None
        self.vocab_type = vocab_type

        self.rhyming_eval = RhymingEval(dataset_identifier = args.data_type) #[SONNET_DATASET_IDENTIFIER, LIMERICK_DATASET_IDENTIFIER]
        if args.data_type == SONNET_ENDINGS_DATASET_IDENTIFIER:
            line_endings_file_location = args.all_line_endings_sonnet
            line_endings_file_location_test = args.all_line_endings_sonnet_test
        elif args.data_type == LIMERICK_DATASET_IDENTIFIER:
            line_endings_file_location = args.all_line_endings_limerick
            line_endings_file_location_test = args.all_line_endings_limerick_test
        self.rhyming_eval.setup(line_endings_file_location, line_endings_file_location_test)
        print(" ----->> self.rhyming_eval: ", self.rhyming_eval)

        if args.data_type == LIMERICK_DATASET_IDENTIFIER:
            if os.path.exists(args.limerick_vocab_path):
                self.limerick_vocab = load_sonnet_vocab(args.limerick_vocab_path)
                print("No. of tokens loaded from existing vocab : ", len(self.limerick_vocab))
            else:
                self.limerick_vocab = None
        else: # sonnet_endings
            if os.path.exists(args.sonnet_vocab_path):
                self.sonnet_vocab = load_sonnet_vocab(args.sonnet_vocab_path)
                print("No. of tokens loaded from existing vocab : ", len(self.sonnet_vocab))
            else:
                self.sonnet_vocab = None

        self.get_splits()
        self.index()

        self.num_lines = NUM_LINES_QUATRAIN
        if args.data_type == LIMERICK_DATASET_IDENTIFIER:
            self.num_lines = NUM_LINES_LIMERICK

        x_start = self.g_indexer.w2idx[utils.START]
        x_end = self.g_indexer.w2idx[utils.END]
        x_unk=-999
        if utils.UNKNOWN in self.g_indexer.w2idx:
            x_unk = self.g_indexer.w2idx[utils.UNKNOWN]

        #### Vanilla LM model initialized
        num_classes = len(self.g_indexer.w2idx)

        if args.data_type == LIMERICK_DATASET_IDENTIFIER:
            self._trun_bptt_mode = True
            self._validation_bptt_mode = False
            line_separator_symbol = "|"
            n_lines_per_sample = NUM_LINES_LIMERICK
            n_lines_to_gen = NUM_LINES_LIMERICK
        else:
            self._trun_bptt_mode = True
            self._validation_bptt_mode = False
            # self._couplet_mode = True # Setting it to True uses vanilla-lm for couplets
            line_separator_symbol = "<eos>"
            n_lines_per_sample = NUM_LINES_SONNET
            n_lines_to_gen = NUM_LINES_QUATRAIN

        self.lm_model = VanillaLM(args,
                               decoder_hidden_size=600,
                               vocab_indexer=self.g_indexer,
                               vocab=self.g_indexer.idx2w,
                               emb_size=100,
                               num_classes=num_classes,
                               start_idx=1,
                               end_idx=2,
                               padding_idx=0,
                               typ='lstm',
                               max_decoding_steps=120,
                               sampling_scheme="first_word",
                               line_separator_symbol=line_separator_symbol,
                               reverse_each_line=True,
                               n_lines_per_sample=n_lines_per_sample,
                               tie_weights=True,
                               dropout_ratio=0.3,
                               phoneme_embeddings_dim=128,
                               encoder_type=None,
                               encoder_input_size=100,
                               encoder_hidden_size=100,
                               encoder_n_layers=1,
                               n_lines_to_gen=n_lines_to_gen)

        self.lm_model.display_params()

        if args.load_gutenberg:
            self.lm_model.load_gutenberg_we(args.load_gutenberg_path, self.g_indexer.idx2w)
    
        if args.use_cuda:
            print("---")
            self.lm_model = self.lm_model.cuda()

        if mode not in ["train_lm"]:

            #### Generator Model initialized
            H = args.H #128
            emsize = args.emsize #128
            self.model = GeneratorModel(H=H, emsize=args.emsize, o_size=self.g_indexer.w_cnt, start_idx=x_start, end_idx=x_end, unk_idx=x_unk, typ=args.model_type, args=args)
            
            #### Rhyming discriminator initialized
            disc_info = {'i_size': None, 'emsize': emsize, 'H': H, 'solver_model_name': None}
            self.disc = CNN(args, info=disc_info)
        
            if args.use_cuda:
                print("---")
                # self.lm_model = self.lm_model.cuda()
                self.model = self.model.cuda()
                self.disc = self.disc.cuda()
                # self.disc_syllable = self.disc_syllable.cuda()
                # self.disc_syllable_line = self.disc_syllable_line.cuda()

            if args.load_gutenberg:
                self.model.load_gutenberg_we(args.load_gutenberg_path, self.g_indexer.idx2w)

            if args.freeze_emb:
                self.model.freeze_emb()

            self.model.display_params()
            self.disc.display_params()


    def get_splits(self):
        # self.all_cmu_items = all_items = list(self.cmu_data.items())
        all_items = list(self.cmu_dict.items())
        if self.typ == SONNET_ENDINGS_DATASET_IDENTIFIER:
            self.get_sonnet_splits(self.args.sonnet_data_path)
            self.splits = self.sonnet_splits
        #@NEW
        elif self.typ == LIMERICK_DATASET_IDENTIFIER:
            self.get_limerick_splits(self.args.limerick_data_path)
            self.splits = self.limerick_splits
        else:
            assert False
            for i in range(11):
                random.shuffle(all_items)
            sz = len(all_items)
            train = all_items[:int(0.8 * sz)]
            val = all_items[int(0.8 * sz):int(0.9 * sz)]
            test = all_items[int(0.9 * sz):]
            self.splits = {'train': train, 'val': val, 'test': test}


    def _split_into_lines(self, txt, split_sym):
        return [line.strip() for line in txt.split(split_sym) if len(line.strip()) > 0]


    def text_to_instance(self, instance_string,
                         line_separator_symbol="<eos>",
                         trun_bptt_mode=True,
                         reverse_each_line=True,
                         n_lines_per_sample=14):
        # pylint: disable=arguments-differ
        fields = dict()
        metadata = {'instance_str': instance_string}
        reverse_target_string = ' '.join(reversed(instance_string.split()))
        tokenized_target = reverse_target_string.split()
        tokenized_target.append(line_separator_symbol)
        source_tokens_lm = tokenized_target[:-1]
        target_tokens_lm = tokenized_target[1:]
        fields['source_tokens_lm'] = source_tokens_lm
        fields['target_tokens_lm'] = target_tokens_lm
        ending_words_mask = [1 if token == line_separator_symbol else 0 for token in tokenized_target[:-1]]
        fields['ending_words_mask'] = np.array(ending_words_mask)
        lines = self._split_into_lines(txt=instance_string, split_sym=line_separator_symbol)
        ending_words = [line.split()[-1] for line in lines]
        if trun_bptt_mode:
            lines = self._split_into_lines(txt=instance_string, split_sym=line_separator_symbol)
            if reverse_each_line:
                lines.reverse()
                ending_words.reverse()

            for i in range(int(n_lines_per_sample/2)):
                if reverse_each_line:
                    line1 = ' '.join(reversed(lines[i * 2].split())) + " " + line_separator_symbol
                    line2 = ' '.join(reversed(lines[i * 2 + 1].split())) + " " + line_separator_symbol
                else:
                    line1 = lines[i*2] + " " + line_separator_symbol
                    line2 = lines[i * 2 + 1] + " " + line_separator_symbol
                line = line1 + " " + line2
                tokenized_target = line.split()
                if reverse_each_line:
                    tokenized_target.insert(0, line_separator_symbol)
                source_tokens_i = tokenized_target[:-1]
                target_tokens_i = tokenized_target[1:]
                if i == 0:
                    ending_words_i = ending_words[0: 2]
                elif i % 2 == 1:
                    ending_words_i = ending_words[i * 2: (i + 2) * 2]
                else:
                    ending_words_i = ending_words[(i - 1) * 2: (i + 1) * 2]
                ending_words_i_textfield = ending_words_i
                fields['source_tokens_'+str(i)] = source_tokens_i
                fields['target_tokens_'+str(i)] = target_tokens_i
                fields['ending_words_'+str(i)] = ending_words_i_textfield

                ending_words_mask_i = [1 if token == line_separator_symbol else 0 for token in tokenized_target[:-1]]

                # In case of couplet mode, unmask first 2 ending words (remember we have reversed lines)
                if _couplet_mode:
                    n_words_to_unmask = 2
                    for z in range(len(ending_words_mask_i)):

                        if ending_words_mask_i[z] == 1:
                            ending_words_mask_i[z] = 0
                            n_words_to_unmask -= 1

                        if n_words_to_unmask == 0:
                            break

                fields['ending_words_mask_'+str(i)] = np.array(ending_words_mask_i)

            reversed_lines = []
            for line in lines:
                reversed_line = line.split()
                reversed_line.reverse()
                reversed_lines.append(reversed_line)
            reversed_quatrains = []
            reversed_quatrains.append(reversed_lines[2:6])
            reversed_quatrains.append(reversed_lines[6:10])
            reversed_quatrains.append(reversed_lines[10:14])
            fields['reversed_quatrains'] = reversed_quatrains
        fields['ending_words_reversed'] = ending_words
        fields['metadata'] = metadata
        return fields


    def _load_sonnet_data(self, sonnet_data_path, split):
        sonnet_data_file = sonnet_data_path + split + '.txt'
        data = open(sonnet_data_file, 'r').readlines()
        ret = []
        skipped = 0
        n_tokens = 0
        n_endings = 0
        for sonnet in data:
            if translator is not None:
                instance_string = sonnet.strip().translate(translator).replace('‘','')
            else:
                instance_string = sonnet.strip().replace('‘', '')
            instance = self.text_to_instance(instance_string=instance_string)
            lines = instance_string.split(' <eos>')[:-1]
            last_words = [line.strip().split()[-1] for line in lines]
            instance['lines'] = lines
            instance['ending_words'] = last_words
            instance['skip'] = False
            n_tokens += len(instance['target_tokens_lm'])
            n_endings += len(last_words)
            for j, w in enumerate(last_words[:4]):
                if w not in self.cmu_dict:
                    skipped += 1
                    instance['skip'] = True
                    break
            # ret.append({'lines': lines, 'ending_words': last_words})
            ret.append(instance)
        print("[_load_sonnet_data] : split=", split, " skipped ", skipped, " out of ", len(data))
        print("No. of tokens in sonnet split :", split, " : ", n_tokens)
        print("No. of endings in sonnet split :", split, " : ", n_endings)

        return ret

    def get_sonnet_splits(self, data_path="../data/sonnet_"):
        if self.sonnet_splits is None:
            self.sonnet_splits = {k: self._load_sonnet_data(data_path, k) for k in ['train', 'valid', 'test']}
            self.sonnet_splits['val'] = self.sonnet_splits['valid']
            # self.sonnet_splits = {k:self._process_sonnet_data(val) for k,val in self.sonnet_splits.items()}


    def text_to_instance_limerick(self, instance_string,
                         line_separator_symbol="|",
                         trun_bptt_mode=True,
                         reverse_each_line=True,
                         n_lines_per_sample=5):
        # pylint: disable=arguments-differ
        fields = dict()

        metadata = {'instance_str': instance_string}

        lines = self._split_into_lines(txt=instance_string, split_sym=line_separator_symbol)
        ending_words = [line.split()[-1] for line in lines]

        if trun_bptt_mode:
            # lines = self._split_into_lines(txt=target_string, split_sym=self.line_separator_symbol)
            if reverse_each_line:
                lines.reverse()
                ending_words.reverse()
            line = ''
            for i in range(n_lines_per_sample):
                if reverse_each_line:
                    line1 = ' '.join(reversed(lines[i].split())) + " " + line_separator_symbol
                else:
                    line1 = lines[i] + " " + line_separator_symbol
                if i != (n_lines_per_sample - 1):
                    line = line + line1 + " "
                else:
                    line = line + line1
            tokenized_target = line.split()
            if reverse_each_line:
                tokenized_target.insert(0, line_separator_symbol)
            source_tokens_i = tokenized_target[:-1]
            target_tokens_i = tokenized_target[1:]
            fields['source_tokens_lm'] = source_tokens_i
            fields['target_tokens_lm'] = target_tokens_i
            ending_words_mask = [1 if token == line_separator_symbol else 0 for token in tokenized_target[:-1]]
            fields['ending_words_mask'] = np.array(ending_words_mask)
            reversed_lines = []
            for line in lines:
                reversed_line = line.split()
                reversed_line.reverse()
                reversed_lines.append(reversed_line)
            fields['reversed_quatrains'] = reversed_lines
        fields['ending_words_reversed'] = ending_words
        fields['metadata'] = metadata
        return fields


    def _load_limerick_data(self, limerick_data_path, split):
        data_file = os.path.join(limerick_data_path, split + '_0.json')
        print("Loading from ", data_file)
        data = json.load(open(data_file, 'r'))
        ret = []
        skipped = 0
        n_tokens = 0
        for limerick in data:
            if translator_limerick is not None:
                limerick = limerick['txt'].strip().translate(translator_limerick)
            else:
                limerick = limerick['txt'].strip()
            lines = limerick.split('|')
            if len(lines) != 5:
                skipped += 1
                continue
            instance = self.text_to_instance_limerick(instance_string=limerick)
            # pdb.set_trace()
            last_words = [line.strip().split()[-1] for line in lines]
            instance['lines'] = lines
            instance['ending_words'] = last_words
            instance['skip'] = False
            n_tokens += len(instance['target_tokens_lm'])
            ret.append(instance)
        print("[_load_sonnet_data] : split=", split, " skipped ", skipped, " out of ", len(data))
        print("No. of tokens in limerick split ", split, " : ", n_tokens)
        return ret


    def get_limerick_splits(self, data_path="../data/limerick_only_subset/"):
        if self.limerick_splits is None:
            self.limerick_splits = {k:self._load_limerick_data(data_path, k) for k in ['train','val','test'] }


    def index(self):  # , typ='g2p'):
        if False:  # typ=="g2p":
            self.g_indexer = Indexer(self.args)
            items = list(self.cmu_dict.items())  # self.all_cmu_items
            self.g_indexer.process([i[0] for i in items])
        elif self.typ=="limerick":
            self.get_limerick_splits()
            self.g_indexer = Indexer(self.args)
            if self.limerick_vocab is not None:
                self.g_indexer.process([self.limerick_vocab])
            elif self.vocab_type == "only_ending_words":
                for split, items in self.limerick_splits.items():
                    self.g_indexer.process([i['ending_words'] for i in items])
            else:
                for split, items in self.sonnet_splits.items():
                    self.g_indexer.process([i['source_tokens_lm'] for i in items])
        else:
            self.get_sonnet_splits()
            self.g_indexer = Indexer(self.args)
            if self.sonnet_vocab is not None:
                self.g_indexer.process([self.sonnet_vocab])
            elif self.vocab_type == "only_ending_words":
                for split, items in self.sonnet_splits.items():
                    self.g_indexer.process([i['ending_words'] for i in items])
            else:
                for split, items in self.sonnet_splits.items():
                    self.g_indexer.process([i['source_tokens_lm'] for i in items])


    def save(self, dump_pre):
        pickle.dump(self.splits, open(dump_pre + 'splits.pkl', 'wb'))
        pickle.dump(self.g_indexer, open(dump_pre + 'g_indexer.pkl', 'wb'))


    def load(self, dump_pre):
        self.splits = pickle.load(open(dump_pre + 'splits.pkl', 'rb'))
        self.g_indexer = pickle.load(open(dump_pre + 'g_indexer.pkl', 'rb'))


    def get_dict_by_phoneme(self):
        data = self.splits['train']
        self.last2_to_words = {}
        for g, p in data:
            last2 = '_'.join(p[-2:])
            if last2 not in self.last2_to_words:
                self.last2_to_words[last2] = []
            self.last2_to_words[last2].append(g)


    def batchify_field(self, data, field, use_indexer=True) -> torch.LongTensor:
        field_data = []
        max_len = 0
        for instance in data:

            field_instance = instance[field]
            if len(field_instance) > max_len:
                max_len = len(field_instance)
            if use_indexer:
                field_data.append(self.g_indexer.w_to_idx(w=field_instance))
            else:
                if isinstance(field_instance, np.ndarray):
                    field_data.append(field_instance.tolist())
                else:
                    field_data.append(field_instance)

        padded_field_data = []
        field_mask = torch.zeros([len(data), max_len], dtype=torch.int32)
        for idx, encoded_field_instance in enumerate(field_data):
            current_len = len(encoded_field_instance)
            field_mask[idx, :current_len] = 1
            # field_mask[:current_len] = encoded_field_instance
            if current_len < max_len:
                padded_field_data.append(np.array(encoded_field_instance+[0]*(max_len - current_len)))
            else:
                padded_field_data.append(np.array(encoded_field_instance))
        # return padded_field_data, field_mask
        # pdb.set_trace()
        return torch.LongTensor(np.array(padded_field_data)), field_mask


    def get_batch_lm(self, i, batch_size, split):

        if self.args.data_type == "limerick":
            data = self.limerick_splits[split][i * batch_size: (i + 1) * batch_size]
        else:
            data = self.sonnet_splits[split][i * batch_size: (i + 1) * batch_size]
        batch = {}
        all_fields = data[0].keys()
        for field in all_fields:
            # pdb.set_trace()
            if field != "metadata" and field != "lines" and field != "skip" and field != "ending_words" and field != "reversed_quatrains":
                if 'mask' in field:
                    field_data, field_mask = self.batchify_field(data=data, field=field,
                                                                 use_indexer=False)
                else:
                    field_data, field_mask = self.batchify_field(data=data, field=field,
                                                                 use_indexer=True)
                batch[field] = {}
                batch[field]["tokens"] = field_data
                batch[field]["mask"] = field_mask
            else:
                batch[field] = [instance[field] for instance in data]
        return batch

    
    def get_stats_ending_words_batch(self, batch):
        if self.args.data_type == "limerick":
            total_limericks = 0
            total_limericks_ending_not_in_vocab = 0
            for ending_words in batch['ending_words']:
                total_limericks += 1
                for word in ending_words:
                    if word not in self.g_indexer.w2idx:
                        total_limericks_ending_not_in_vocab += 1
                        break
            return total_limericks, total_limericks_ending_not_in_vocab
        else:
            total_quatrains = 0
            total_quatrains_ending_not_in_vocab = 0
            for ending_words in batch['ending_words']:
                total_quatrains += 3
                for word in ending_words[0:4]:
                    if word not in self.g_indexer.w2idx:
                        total_quatrains_ending_not_in_vocab += 1
                        break
                for word in ending_words[4:8]:
                    if word not in self.g_indexer.w2idx:
                        total_quatrains_ending_not_in_vocab += 1
                        break
                for word in ending_words[8:12]:
                    if word not in self.g_indexer.w2idx:
                        total_quatrains_ending_not_in_vocab += 1
                        break
            return total_quatrains, total_quatrains_ending_not_in_vocab

        
    def get_stats_ending_words(self, split, batch_size):
        if self.args.data_type == "limerick":
            num_batches = self.get_num_batches(split=split, batch_size=batch_size, data_type="limerick")
            total_limericks = 0
            total_limericks_ending_not_in_vocab = 0
            for batch_idx in range(num_batches):
                batch = self.get_batch_lm(i=batch_idx, batch_size=32, split='train')
                limericks_batch, limericks_ending_not_in_vocab_batch = self.get_stats_ending_words_batch(batch=batch)
                total_limericks += limericks_batch
                total_limericks_ending_not_in_vocab += limericks_ending_not_in_vocab_batch
            print("No. of quatrains : ", total_limericks)
            print("No. of quatrains with endings in vocab : ", total_limericks - total_limericks_ending_not_in_vocab)
        else:
            num_batches = self.get_num_batches(split=split, batch_size=batch_size, data_type="sonnet")
            total_quatrains = 0
            total_quatrains_ending_not_in_vocab = 0
            for batch_idx in range(num_batches):
                batch = self.get_batch_lm(i=batch_idx, batch_size=32, split='train')
                quatrains_batch, quatrains_ending_not_in_vocab_batch = self.get_stats_ending_words_batch(batch=batch)
                total_quatrains += quatrains_batch
                total_quatrains_ending_not_in_vocab += quatrains_ending_not_in_vocab_batch
            print("No. of quatrains : ", total_quatrains)
            print("No. of quatrains with endings in vocab : ", total_quatrains - total_quatrains_ending_not_in_vocab)

            
    def endings_in_vocab(self, ending_words):
        for word in ending_words:
            if word not in self.g_indexer.w2idx:
                return False
        return True

    
    def get_batch(self, i, batch_size, split, last_two=False, add_end_to_y=True, skip_unk=True):  # typ='g2p',
        # typ: g2p, ae
        # assert typ==self.typ
        typ = self.typ
        if typ == "sonnet_endings":
            data = self.sonnet_splits[split][i * batch_size:(i + 1) * batch_size]
            x = []
            for val in data:
                if self.args.use_all_sonnet_data:
                    for st in [0,4,8]:
                        if (not skip_unk) or self.endings_in_vocab(ending_words=val['ending_words'][st:st+4]):
                            x_val = self.prepare_sonnet_x(val['ending_words'][st:st+4])
                            x.append(x_val['indexed_x'])
                else:
                    if (not skip_unk) or self.endings_in_vocab(ending_words=val['ending_words'][0:4]):
                        x_val = self.prepare_sonnet_x(val['ending_words'][0:4])
                        x.append(x_val['indexed_x'])
                    elif (not skip_unk) or self.endings_in_vocab(ending_words=val['ending_words'][4:8]):
                        x_val = self.prepare_sonnet_x(val['ending_words'][4:8])
                        x.append(x_val['indexed_x'])
                    elif (not skip_unk) or self.endings_in_vocab(ending_words=val['ending_words'][8:12]):
                        x_val = self.prepare_sonnet_x(val['ending_words'][8:12])
                        x.append(x_val['indexed_x'])
                    else:
                        continue
            x_end = self.g_indexer.w2idx[utils.END]
            x = [x_i + [x_end] for x_i in x]
            x_start = self.g_indexer.w2idx[utils.START]
            return x, None, x_start
        elif typ=="limerick":
            data = self.limerick_splits[split][i*batch_size:(i+1)*batch_size]
            x = []
            for val in data:
                if (not skip_unk) or self.endings_in_vocab(ending_words=val['ending_words']):
                    x_val = self.prepare_sonnet_x(val['ending_words'])
                    x.append(x_val['indexed_x'])
                else:
                    continue
            x_end = self.g_indexer.w2idx[utils.END]
            x = [x_i+[x_end] for x_i in x]
            x_start = self.g_indexer.w2idx[utils.START]
            return x,None,x_start
        else:
            assert False


    def get_num_batches(self, split, batch_size, data_type=None):
        if data_type == "limerick":
            return int((len(self.limerick_splits[split]) + batch_size - 1.0) / batch_size)
        elif data_type == "sonnet":
            return int((len(self.sonnet_splits[split]) + batch_size - 1.0) / batch_size)
        else:
            return int((len(self.splits[split]) + batch_size - 1.0) / batch_size)


    def prepare_sonnet_x(self, lst_of_words):
        ret = lst_of_words
        ret_idx = []
        for w in ret:
            if w not in self.g_indexer.w2idx:
                w = UNKNOWN
            ret_idx.append( self.g_indexer.w2idx[w] )
        return {'x': ret, 'indexed_x': ret_idx}

    ################

    def batch_loss_lm_2lines(self, batch, for_training, batch_idx=None, decoder_hidden=None, decoder_context=None, hier_mode=False) -> (torch.Tensor, Dict[str, torch.Tensor]):

        if batch_idx is None:
            output_dict = self.lm_model(source_tokens=batch["source_tokens_lm"],
                                     target_tokens=batch["target_tokens_lm"],
                                     ending_words=batch["ending_words_reversed"],
                                     # metadata=batch["metadata"],
                                     batch_idx=batch_idx,
                                     decoder_hidden=decoder_hidden,
                                     decoder_context=decoder_context,
                                     ending_words_mask=batch["ending_words_mask"],
                                     hier_mode = hier_mode)
        else:
            output_dict = self.lm_model(source_tokens=batch["source_tokens_" + str(batch_idx)],
                                     target_tokens=batch["target_tokens_" + str(batch_idx)],
                                     ending_words=batch["ending_words_" + str(batch_idx)],
                                     # metadata=batch["metadata"],
                                     batch_idx=batch_idx,
                                     decoder_hidden=decoder_hidden,
                                     decoder_context=decoder_context,
                                     ending_words_mask=batch["ending_words_mask_" + str(batch_idx)], hier_mode=hier_mode)

        try:
            loss = output_dict["loss"]
            # if for_training:
            #     loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None
        return loss, output_dict


    def get_batch_loss_lm(self, split, batch, batch_size, optimizer, mode='train'):

        n_batches = 0
        train_loss = 0
        batch = self.get_batch_lm(i=batch, batch_size=batch_size, split=split)
        if self.args.use_cuda:
            batch = {k: (
                {k1: (v1.cuda() if isinstance(v1, torch.Tensor) and torch.cuda.is_available() else v1) for k1, v1 in v.items()} if isinstance(v,
                                                                                                                dict) else v)
                     for k, v in batch.items()}
        decoder_hidden = None
        decoder_context = None

        if self.args.data_type=="limerick":
            n_batches += 1
            optimizer.zero_grad()
            loss, output_dict = self.batch_loss_lm_2lines(batch=batch, for_training=True, batch_idx=None,
                                                          decoder_hidden=decoder_hidden,
                                                          decoder_context=decoder_context)
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        else:
            for i in range(7):
                n_batches += 1
                optimizer.zero_grad()
                loss, output_dict = self.batch_loss_lm_2lines(batch=batch, for_training=True, batch_idx=i,
                                                              decoder_hidden=decoder_hidden,
                                                              decoder_context=decoder_context)
                # loss = self.batch_loss(batch_group, for_training=True)
                # pdb.set_trace()
                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")
                loss.backward()
                train_loss += loss.item()
                # batch_grad_norm = self.rescale_gradients()
                optimizer.step()
                if self._trun_bptt_mode:
                    decoder_hidden = torch.stack(output_dict["decoder_hidden"]).detach()
                    decoder_context = torch.stack(output_dict["decoder_context"]).detach()

        return train_loss, n_batches


    def train_epoch_lm(self, epoch, optimizer, batch_size, split='train') -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        train_loss = 0.0

        # Shuffle training data
        if self.args.data_type == "limerick":
            random.shuffle(self.limerick_splits[split])
        else:
            random.shuffle(self.sonnet_splits[split])

        # Set the model to "train" mode.
        self.lm_model.train()

        last_save_time = time.time()
        if self.args.data_type == LIMERICK_DATASET_IDENTIFIER:
            num_batches = self.get_num_batches(split=split, batch_size=batch_size, data_type="limerick")
        else:
            num_batches = self.get_num_batches(split=split, batch_size=batch_size, data_type="sonnet")

        batches_this_epoch = 0

        print("Training epoch : ", epoch)

        # cumulative_batch_size = 0

        # Get tqdm for the training batches
        train_generator_tqdm = Tqdm.tqdm(range(num_batches),
                                         total=num_batches)
        # for batch_idx in tqdm(range(num_batches)):
        for batch_idx in train_generator_tqdm:
                train_loss_batch, n_batches = self.get_batch_loss_lm(split=split, batch=batch_idx,
                                                                     batch_size=batch_size, optimizer=optimizer)
                train_loss += train_loss_batch
                batches_this_epoch += n_batches
                # Update the description with the latest metrics
                metrics = get_metrics(train_loss, batches_this_epoch)
                description = description_from_metrics(metrics)
                train_generator_tqdm.set_description(description, refresh=False)
        metrics = get_metrics(train_loss, batches_this_epoch)
        return metrics


    def validation_loss_lm(self, split, batch_size=64, hier_mode=False) -> Tuple[float, int, int, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        print("Evaluating : ", split)
        self.lm_model.eval()
        if self.args.data_type == LIMERICK_DATASET_IDENTIFIER:
            num_batches = self.get_num_batches(split=split, batch_size=batch_size, data_type="limerick")
        else:
            num_batches = self.get_num_batches(split=split, batch_size=batch_size, data_type="sonnet")
        # Get tqdm for the validation batches
        val_generator_tqdm = Tqdm.tqdm(range(num_batches),
                                       total=num_batches)
        batches_this_epoch = 0
        val_loss = 0
        n_words = 0
        n_samples = 0
        for batch_idx in val_generator_tqdm:
            batch = self.get_batch_lm(i=batch_idx, batch_size=batch_size, split=split)
            if self.args.use_cuda:
                batch = {k: (
                    {k1: (v1.cuda() if isinstance(v1, torch.Tensor) and torch.cuda.is_available() else v1) for k1, v1 in v.items()} if isinstance(v,
                                                                                                                    dict) else v)
                         for k, v in batch.items()}
            decoder_hidden = None
            decoder_context = None
            loss_this_batch = None
            lengths_this_batch = None

            if self.args.data_type == LIMERICK_DATASET_IDENTIFIER:
                # Computing batch-loss for all 5 reversed limerick lines
                loss, output_dict = self.batch_loss_lm_2lines(batch, for_training=False, hier_mode=hier_mode)

                lengths_this_batch = np.array([len.detach().cpu().numpy() for len in
                                               output_dict['target_sentence_lengths']])
                if hier_mode:
                    lengths_this_batch -= 5

                loss_this_batch = np.array([_loss.detach().cpu().numpy() * len for _loss, len in
                                            zip(output_dict['loss'], lengths_this_batch)])

            else:
                # Compute loss for sonnet dataset
                if self._validation_bptt_mode:
                    # Computing batch-loss every 2 reversed sonnet lines
                    for i in range(7):

                        loss, output_dict = self.batch_loss_lm_2lines(batch=batch, for_training=False, batch_idx=i,
                                                            decoder_hidden=decoder_hidden, decoder_context=decoder_context, hier_mode=hier_mode)

                        lengths_i = np.array([len.detach().cpu().numpy() for len in
                                     output_dict['target_sentence_lengths']])

                        # If we are computing loss/ppl for hierarchical model drop two ending words (Note that while computing loss this is taken into account by target_mask)
                        if hier_mode:
                            lengths_i -= 2

                        loss_i = np.array([_loss.detach().cpu().numpy() * len for _loss, len in zip(output_dict['loss'], lengths_i)])

                        if loss_this_batch is None:
                            lengths_this_batch = lengths_i
                            loss_this_batch = loss_i
                        else:
                            lengths_this_batch += lengths_i
                            loss_this_batch += loss_i

                        decoder_hidden = torch.stack(output_dict["decoder_hidden"]).detach()
                        decoder_context = torch.stack(output_dict["decoder_context"]).detach()

                else:
                    # Computing batch-loss for all 14 reversed sonnet lines
                    loss, output_dict = self.batch_loss_lm_2lines(batch, for_training=False, hier_mode=hier_mode)

                    lengths_this_batch = np.array([len.detach().cpu().numpy() for len in
                                 output_dict['target_sentence_lengths']])
                    if hier_mode and _couplet_mode:
                        lengths_this_batch -= 12
                    elif hier_mode:
                            lengths_this_batch -= 14

                    loss_this_batch = np.array([_loss.detach().cpu().numpy() * len for _loss, len in zip(output_dict['loss'], lengths_this_batch)])

            loss = np.sum(loss_this_batch)
            if loss is not None:
                batches_this_epoch += 1
                val_loss += loss
                n_words += np.sum(lengths_this_batch)
                n_samples += len(lengths_this_batch)

            # Update the description with the latest metrics
            val_metrics = get_metrics(val_loss, n_words)
            description = description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

            print("Total Loss : ", val_loss, " | no. of words : ", n_words, " | samples : ", n_samples, " | PPL : ",
                  np.exp(val_loss / n_words))

        return val_loss, batches_this_epoch, n_words, n_samples


    def shuffle_data(self, split='train'):

        print("Data shuffled based upon source_tokens_lm!")
        # Shuffle training data
        instance_len = [len(instance["source_tokens_lm"]) for instance in self.sonnet_splits[split]]
        sorted_indices = sorted(range(len(instance_len)), key=lambda k: instance_len[k])

        self.sonnet_splits[split] = [self.sonnet_splits[split][index] for index in sorted_indices]


    def train_lm(self, epochs=25, debug=True, args=None):

        learning_rate = 0.0025
        optimizer = torch.optim.Adam(self.lm_model.parameters(), lr=learning_rate)
        best_loss = 999999999999.0
        best_lm_model = self.lm_model
        best_epoch = 0
        batch_size = 32

        training_start_time = time.time()

        all_loss_tracker = {}
        all_loss_tracker_val = {}

        # Shuffle training data before training
        # self.shuffle_data(split='train')

        for epoch in range(epochs):

            epoch_start_time = time.time()
            train_metrics = self.train_epoch_lm(epoch, optimizer, batch_size)

            print("Validation split!")
            with torch.no_grad():
                # We have a validation set, so compute all the metrics on it.
                val_loss, num_batches, n_words, n_samples = self.validation_loss_lm(split='val', batch_size=64)

                val_metrics = get_metrics(val_loss, n_words)

                # Check validation metric for early stopping
                # this_epoch_val_metric = val_metrics[self._validation_metric]
                # self._metric_tracker.add_metric(this_epoch_val_metric)
                #
                # if self._metric_tracker.should_stop_early():
                #     logger.info("Ran out of patience.  Stopping training.")
                #     break

            print("Test split!")

            with torch.no_grad():
                # We have a validation set, so compute all the metrics on it.
                test_loss, num_batches, n_words, n_samples = self.validation_loss_lm(split='test', batch_size=64)

                test_metrics = get_metrics(test_loss, n_words)

            # if debug:
                # break

            print("TRAIN epoch perplexity = ", np.exp(train_metrics['loss']))
            print()

            print("VAL perplexity = ", np.exp(val_metrics['loss']))
            print("TEST perplexity = ", np.exp(test_metrics['loss']))

            if val_metrics['loss'] < best_loss:
                best_epoch = epoch
                if args.save_vanilla_lm:
                    torch.save(self.lm_model.state_dict(), os.path.join(self.args.vanilla_lm_path, 'model_' + str(epoch)))
                    print("Best LM model until epoch ", epoch, "dumped!")

                best_loss = val_metrics['loss']

        print("Best model from epoch ", best_epoch, " saved!")
        best_lm_model_state_dict = torch.load(os.path.join(self.args.vanilla_lm_path, 'model_' + str(best_epoch)))
        torch.save(best_lm_model_state_dict, os.path.join(self.args.vanilla_lm_path, 'model_best'))


    def get_loss(self, split, batch, batch_size, mode='train', criterion=None):

        model = self.model
        if mode == "train":
            model.train()
        else:
            model.eval()
        len_output = 0
        # x, _, x_start = self.get_batch(i=batch, batch_size=batch_size, split=split)  # , typ=args.data_type)
        x, _, x_start = self.get_batch(i=batch, batch_size=batch_size, split=split, skip_unk=False)  # , typ=args.data_type)

        batch_loss = torch.tensor(0.0)
        if self.args.use_cuda:
            batch_loss = batch_loss.cuda()

        i = 0
        all_e = []
        # print(" -- batch=",batch, " || x: ",len(x))
        for x_i in x:
            len_output += len(x_i)
            info = model(x_i, use_gt=True)
            out_all_i = info['out_all']
            i += 1
            # print(" *** out_all_i = ", out_all_i)
            out_all_i = torch.stack(out_all_i)
            dist = out_all_i.view(-1, self.g_indexer.w_cnt)
            targets = np.array(x_i, dtype=np.long)
            targets = torch.from_numpy(targets)
            if self.args.use_cuda:
                targets = targets.cuda()
            cur_loss = criterion(dist, targets)
            # print(cur_loss)
            batch_loss += cur_loss

        total_loss = batch_loss
        return total_loss, len_output, {
            'total_batch_loss': total_loss.data.cpu().item(), \
            'batch_recon_loss': batch_loss.data.cpu().item(), \
            'elbo_loss': batch_loss.data.cpu().item()
        }


    def train(self, epochs=11, debug=False, train_lm_supervised=False):

        print("train_lm_supervised = ", train_lm_supervised)

        batch_size = 32

        # Ending words generator model
        model = self.model
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = 999999999999.0
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Vanilla-LM (Conditional quatrain generator model)

        learning_rate = 0.001
        optimizer_lm = torch.optim.Adam(self.lm_model.parameters(), lr=learning_rate)

        # Rhyming Discriminator
        learning_rate = 1e-4
        optimizer_disc = torch.optim.Adam(self.disc.parameters(), lr=learning_rate)

        # Syllable Discriminator
        learning_rate = 1e-4
        # optimizer_disc_syllable = torch.optim.Adam(self.disc_syllable.parameters(), lr=learning_rate)
        learning_rate = 1e-4
        # optimizer_disc_syllable_line = torch.optim.Adam(self.disc_syllable_line.parameters(), lr=learning_rate)

        self.train_summary = []

        for epoch in range(epochs):

            epoch_summary = {'epoch':epoch}

            # Enable train() mode
            model.train()
            self.lm_model.train()

            print("Training epoch : ", epoch)

            # Shuffle training data
            # random.shuffle(self.sonnet_splits['train'])
            #@NEW
            random.shuffle(self.splits['train'])

            all_loss_tracker = {}
            all_loss_tracker_val = {}
            all_loss_tracker_disc = {}
            dump_info_all = []

            num_batches = self.get_num_batches('train', batch_size)
            epoch_loss = 0.0
            train_lm_loss = 0.0
            ctr = 0
            data_point_ctr = 0
            data_point_ctr_syll_gen = 0
            data_point_ctr_syll_gen_line = 0
            data_point_ctr_syll_train = 0
            data_point_ctr_syll_train_line = 0

            # Get tqdm for the training batches
            train_generator_tqdm = Tqdm.tqdm(range(num_batches),
                                             total=num_batches)
            batches_this_epoch = 0

            for batch in train_generator_tqdm:
            # for batch in range(num_batches):


                ############ SUPERVISED ENDING WORD GENERATOR
                total_batch_loss, len_output, info = self.get_loss('train', batch, batch_size, mode='train',
                                                                   criterion=criterion)
                # print("batch_loss = ", batch_loss/len(x_i))
                if debug:
                    print("TRAIN info = ", info)
                for k, v in info.items():
                    if k.count('loss') > 0:
                        if k not in all_loss_tracker:
                            all_loss_tracker[k] = 0.0
                        all_loss_tracker[k] += v
                epoch_loss = epoch_loss + total_batch_loss.data.cpu().item()

                model.zero_grad()
                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()

                ctr = ctr + len_output
                if batch % 1000 == 0:
                    print("TRAIN batch = ", batch, "epoch_loss/ctr = ", epoch_loss / ctr)
                # metrics = get_metrics(epoch_loss, ctr, key="endings_supervised_loss")


                ############ SUPERVISED LM
                if train_lm_supervised:
                    train_lm_loss_batch, n_batches = self.get_batch_loss_lm(split='train', batch=batch,
                                                                         batch_size=batch_size, optimizer=optimizer_lm)
                    train_lm_loss += train_lm_loss_batch
                    batches_this_epoch += n_batches

                    print("Supervised LM train_lm_loss/batches_this_epoch = ", train_lm_loss/batches_this_epoch)
                    # metrics = get_metrics(train_lm_loss, batches_this_epoch, key="lm_loss", metrics=metrics)

                # print("dfdf-----exitiiiiiii")
                # Update tqdm description with 'gen_loss' and 'lm_loss'
                # description = description_from_metrics(metrics)
                # train_generator_tqdm.set_description(description, refresh=False)


                ############# STRUCTURE

                if self.args.use_reinforce_loss:

                    model.eval()
                    # x, _, x_start = self.get_batch(i=batch, batch_size=batch_size, split='train')  # , typ=args.data_type)
                    x, _, x_start = self.get_batch(i=batch, batch_size=batch_size, split='train', skip_unk=True)  # , typ=args.data_type)
                    # lm_batch_info = self.get_batch_for_lm(i=batch, batch_size=batch_size, split='train') #, typ=args.data_type)
                    data_point_ctr += len(x)
                    all_line_endings_gen = []
                    all_line_endings_train = []
                    all_rewards = []
                    all_rewards_syll = []
                    all_rewards_syll_line = []
                    all_log_probs = []
                    all_log_probs_syll = []
                    all_log_probs_syll_line = []

                    if train_lm_supervised:
                        lm_batch = self.get_batch_lm(i=batch, batch_size=batch_size, split='train')
                        assert False, "TODO"
                        # print(" --- lm_batch = ", lm_batch.keys())
                        # batchsize * 3 * 4
                        # print(" --- lm_batch[reversed_quatrains] = ", len(lm_batch['reversed_quatrains']))
                        # print(" --- lm_batch[reversed_quatrains] = ", len(lm_batch['reversed_quatrains'][0]))
                        # print(" --- lm_batch[reversed_quatrains] = ", len(lm_batch['reversed_quatrains'][0][0]))
                        # print(" --- lm_batch[reversed_quatrains] = ", lm_batch['reversed_quatrains'][0][0])

                    # print("--- REINFORCE ---")
                    pr = False  # Random prints
                    if debug or np.random.rand() < 0.05:
                        pr = True
                    # pr=False
                    disc_loss_batch = 0.0
                    disc_loss_batch_line = 0.0

                    for i in range(len(x)):

                        #####----------------- ENDINGS
                        info = model(gt=None, use_gt=False)  # Get Sample
                        # print("info['actions'] = ", info['actions'])
                        # print(self.g_indexer.idx_to_2( [i.data.cpu().item() for i in info['actions']] ))
                        line_endings_gen = info[
                            'actions']  # [i.data.cpu().item() for i in info['actions']] #info['actions']
                        line_endings_train = []
                        all_line_endings_gen.append(line_endings_gen)

                        # for w in x[i][:4]:
                        #@NEW
                        #print("x[i] = ",x[i])
                        for w in x[i][:-1]: # just removing end?
                            word_idx = torch.tensor(w)
                            if self.args.use_cuda:
                                word_idx = word_idx.cuda()
                            line_endings_train.append(word_idx)
                            ### TODO: do not use unks? -- Currently endingwords are returned fro a quarttain only if all ending words are in dictionary. so it is fine to ignore for now. just change in sampling part though
                        all_line_endings_train.append(line_endings_train)

                        try:
                            # if True:
                            disc_info = self.disc.update_discriminator(line_endings_gen, line_endings_train, pr,
                                                                       self.g_indexer.idx2w)
                            # print("disc_info=", disc_info)
                            self.disc.zero_grad()
                            optimizer_disc.zero_grad()
                            if pr:
                                dump_info_all.append(disc_info['dump_info'])
                            disc_loss = disc_info['loss']
                            if 'disc_loss' not in all_loss_tracker_disc:
                                all_loss_tracker_disc['disc_loss'] = 0.0
                                all_loss_tracker_disc['reward'] = 0.0
                            all_loss_tracker_disc['disc_loss'] += disc_loss.data.cpu().item()
                            # disc_loss.backward(retain_graph=True)
                            disc_loss.backward()
                            optimizer_disc.step()

                            reward = disc_info['reward']
                            all_loss_tracker_disc['reward'] += reward.data.cpu().item()
                            if debug:
                                print("[rhyming] ----> reward = ", reward)
                            all_rewards.append(reward.data.cpu().item())
                            # print("info['logprobs'] = ", info['logprobs'])
                            log_probs = torch.stack(info['logprobs'])
                            all_log_probs.append(log_probs)
                            if debug:
                                print("[[rhyming] ][reinforce] log_probs =", log_probs)
                        except:
                           print("--[[rhyming] ] exception--")  ##TODO: Need to fix what causes these issues
                        #    # print(" exception = ", e)
                        if debug:
                            print()

                ## Disc: Rhyming reinforce updates to Generator
                if self.args.use_reinforce_loss:
                    all_rewards_numpy = np.array(all_rewards)
                    assert len(all_rewards_numpy) == len(all_log_probs)
                    # print("all_rewards_numpy = ", all_rewards_numpy)
                    all_rewards_numpy = (all_rewards_numpy - all_rewards_numpy.mean()) / (all_rewards_numpy.std())
                    # print("all_rewards_numpy = ", all_rewards_numpy)
                    ##### -- TODO: this is inefficent. just compute for every and then backpropr
                    for q, log_probs in enumerate(all_log_probs):
                        loss = -torch.sum(log_probs * all_rewards_numpy[q])
                        if pr:
                            print("[reinforce] loss =", loss)
                        loss = loss * self.args.reinforce_weight
                        model.zero_grad()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                ########## WITHIN EPOCH PRINTS AND OPTIONS
                if batch % 1000 == 0:
                    if 'disc_loss' in all_loss_tracker_disc:
                        print("TRAIN batch = ", batch, "all_loss_tracker_disc[disc_loss]/data_point_ctr = ",
                            all_loss_tracker_disc['disc_loss'] / data_point_ctr)
                    if 'disc_loss_syll' in all_loss_tracker_disc:
                        print("TRAIN batch = ", batch, "all_loss_tracker_disc['disc_loss_syll']/normalizert = ",
                            all_loss_tracker_disc['disc_loss_syll'] / (data_point_ctr_syll_gen +data_point_ctr_syll_train) )

                if debug:
                    break

            ################### EPOCH END OPTIONS
            # print("epoch_loss = ", epoch_loss/ctr)
            print("TRAIN epoch perplexity = ", np.exp(all_loss_tracker['elbo_loss'] / ctr))
            epoch_summary.update({'train_elbo_loss':all_loss_tracker['elbo_loss'] , 
                'train_ppl':np.exp(all_loss_tracker['elbo_loss'] / ctr), 
                'ctr':ctr})
            print("TRAIN epoch all_loss_tracker (norm. by num_batches) = ",
                  {k: v / num_batches for k, v in all_loss_tracker.items()})
            all_loss_tracker.update({'ctr': ctr})
            if 'disc_loss' in all_loss_tracker_disc:
                print("TRAIN all_loss_tracker_disc[disc_loss]/data_point_ctr = ",
                    all_loss_tracker_disc.get('disc_loss',-999) / data_point_ctr)
                all_loss_tracker_disc.update({'data_point_ctr': data_point_ctr})
            print()

            ### RUN ON VALIDATION
            # Will skop this when using gan? Only when using inital noise z
            num_batches = self.get_num_batches('val', batch_size)
            epoch_val_loss = 0.0
            ctr = 0
            for batch in range(num_batches):
                total_batch_loss, len_output, info = self.get_loss('val', batch, batch_size, mode='eval',
                                                                   criterion=criterion)
                # print("batch_loss = ", batch_loss/len(x_i))
                if debug:
                    print("VAL info = ", info)
                epoch_val_loss = epoch_val_loss + total_batch_loss.data.cpu().item()
                for k, v in info.items():
                    if k.count('loss') > 0:
                        if k not in all_loss_tracker_val:
                            all_loss_tracker_val[k] = 0.0
                        all_loss_tracker_val[k] += v
                # model.zero_grad()
                # optimizer.zero_grad()
                # total_batch_loss.backward()
                # optimizer.step()
                ctr = ctr + len_output
                if debug:
                    # print(" ---- all_loss_tracker_val = ", all_loss_tracker_val)
                    break
            print("epoch VAL epoch_val_loss  = ", epoch_val_loss, " ctr = ", ctr)
            if 'elbo_loss' in all_loss_tracker_val:
                print("epoch VAL perplexity = ", np.exp(all_loss_tracker_val['elbo_loss'] / ctr))
            all_loss_tracker_val.update({'ctr': ctr})
            epoch_summary.update({'val_elbo_loss':all_loss_tracker_val['elbo_loss'], 
                'train_ppl':np.exp(all_loss_tracker_val['elbo_loss'] / ctr), 
                'ctr':ctr})
            print("epoch all_loss_tracker_val (norm. by num_batches) = ",
                  {k: v / num_batches for k, v in all_loss_tracker_val.items()})
            # if not debug:
            # torch.save(model.state_dict(), self.args.model_dir + 'model_' + str(epoch%5))
            torch.save(model.state_dict(), self.args.model_dir + 'model_' + str(epoch))
            torch.save(self.lm_model.state_dict(), self.args.model_dir + 'lmmodel_' + str(epoch))
            torch.save(self.disc.state_dict(), self.args.model_dir + 'disc_' + str(epoch))
            # torch.save(self.disc_syllable.state_dict(), self.args.model_dir + 'disc_syllable_' + str(epoch))
            # torch.save(self.disc_syllable_line.state_dict(), self.args.model_dir + 'disc_syllable_line_' + str(epoch))
            if (epoch_val_loss / ctr) < best_loss:  ## TODO: We should probably do this by lm model loss
                best_loss = epoch_val_loss / ctr
                torch.save(model.state_dict(), self.args.model_dir + 'model_best')
                torch.save(self.lm_model.state_dict(), self.args.model_dir + 'lmmodel_best')
                torch.save(self.disc.state_dict(), self.args.model_dir + 'disc_best')
                # torch.save(self.disc_syllable.state_dict(), self.args.model_dir + 'disc_syllable_best')
                # torch.save(self.disc_syllable_line.state_dict(), self.args.model_dir + 'disc_syllable_line_best')

            ##### GET ENDINGS SAMPLES
            model.eval()
            samples = []
            self.lm_model.eval() #TODO:  is it turned back to train() ?
            samples_lm = []
            samples_lm_traincond = []
            print("--- ENDINGS Samples ---")
            x, _, x_start = self.get_batch(i=0, batch_size=331, split='val', skip_unk=True)  # , typ=args.data_type)
            random.shuffle(x)
            random.shuffle(x)
            for i in range(self.args.num_samples_at_epoch_end):
                info = model(gt=None, use_gt=False)
                # print("info['actions'] = ", info['actions'])
                sample_str = self.g_indexer.idx_to_2([i.data.cpu().item() for i in info['actions']])
                # print(sample_str)
                samples.append(sample_str)

                ending_words_str = sample_str
                ending_words = info['actions']
                if self.args.use_cuda:
                    ending_words = [e.cuda() for e in ending_words]
                if len(ending_words)!=self.num_lines:
                    sample_str='--endingWordsCount !=4 -- '
                else:
                    ending_words_reversed = [w for w in reversed(ending_words)]
                    info = self.lm_model._sample(ending_words=ending_words_reversed)
                    # info = self.lm_model._sample(ending_words=ending_words)
                    quatrain_gen = info['actions']  # [i.data.cpu().item() for i in info['actions']] #info['actions']
                    sample_str = ' <eos> '.join([ ' '.join([self.g_indexer.idx2w[widx.data.cpu().item()] for widx in reversed(line)]) for line in quatrain_gen ])
                samples_lm.append([ending_words_str, sample_str])
                # print(sample_str)
                line_endings_train = []
                for w in x[i][:-1]: # just removing end?
                    word_idx = torch.tensor(w)
                    if self.args.use_cuda:
                        word_idx = word_idx.cuda()
                    line_endings_train.append(word_idx)
                ending_words = line_endings_train
                ending_words_str = [self.g_indexer.idx2w[widx.data.cpu().item()] for widx in ending_words]
                if len(ending_words)!=self.num_lines:
                    sample_str='--endingWordsCount !=4 -- '
                else:
                    ending_words_reversed = [w for w in reversed(ending_words)]
                    info = self.lm_model._sample(ending_words=ending_words_reversed)
                    quatrain_gen = info['actions']  # [i.data.cpu().item() for i in info['actions']] #info['actions']
                    sample_str = ' <eos> '.join([ ' '.join([self.g_indexer.idx2w[widx.data.cpu().item()] for widx in reversed(line)]) for line in quatrain_gen ])
                # print(sample_str)
                samples_lm_traincond.append([ending_words_str, sample_str])
            
            #### TODO:
            ## compute rhyming and pattern stats
            
            ### TODO: reorg code along 3 groupings
            ## all data loading stuff
            ## all batch related stuff
            ## all training stuff

            if self.args.data_type == LIMERICK_DATASET_IDENTIFIER:
                sonnet_vocab = self.limerick_vocab
            else:
                sonnet_vocab = self.sonnet_vocab
            all_embs = self._get_disc_word_representation_dictionary(sonnet_vocab)
            eval_info = self.rhyming_eval.analyze_embeddings_for_rhyming_from_dict(all_embs)
            thresh_f1, maxf1, test_f1 = eval_info['thresh_f1'], eval_info['maxf1'], eval_info['test_f1']
            print("[EPOCH] = ", epoch, " ---->> thresh_f1, maxf1, test_f1 = ", thresh_f1, maxf1, test_f1)
            epoch_summary.update({'thresh_f1':thresh_f1, 'maxf1':maxf1, 'test_f1':test_f1})

            pattern_eval_info = self.rhyming_eval.analyze_samples_from_endings(samples)
            print("[EPOCH] = ", epoch, " ---->> pattern_eval_info = ", pattern_eval_info) #pattern_success_ratio
            epoch_summary.update({'pattern_eval_info':pattern_eval_info})

            if not os.path.exists(self.args.model_dir + 'samples/'):
                os.makedirs(self.args.model_dir + 'samples/')
            if not os.path.exists(self.args.model_dir + 'samples_lm/'):
                os.makedirs(self.args.model_dir + 'samples_lm/')
            if not os.path.exists(self.args.model_dir + 'logs/'):
                os.makedirs(self.args.model_dir + 'logs/')
            if not os.path.exists(self.args.model_dir + 'dump_info_all/'):
                os.makedirs(self.args.model_dir + 'dump_info_all/')
            json.dump(samples, open(self.args.model_dir + 'samples/' + str(epoch) + '.json', 'w'))
            json.dump(samples_lm, open(self.args.model_dir + 'samples_lm/' + str(epoch) + '.json', 'w'))
            json.dump(samples_lm, open(self.args.model_dir + 'samples_lm/' + str(epoch) + '_traincond.json', 'w'))
            json.dump(all_loss_tracker_val,
                      open(self.args.model_dir + 'logs/all_loss_tracker_val' + str(epoch) + '.json', 'w'))
            json.dump(all_loss_tracker, open(self.args.model_dir + 'logs/all_loss_tracker' + str(epoch) + '.json', 'w'))
            json.dump(all_loss_tracker_disc,
                      open(self.args.model_dir + 'logs/all_loss_tracker_disc' + str(epoch) + '.json', 'w'))
            # print("dump_info_all = ", dump_info_all)
            pickle.dump(dump_info_all,
                        open(self.args.model_dir + 'dump_info_all/dump_info_all' + str(epoch) + '.pkl', 'wb'))
            # if debug:
            #    break
            print()

            self.train_summary.append(epoch_summary)
        
        json.dump(self.train_summary, open(self.args.model_dir + 'train_summary' + '.json', 'w'))
        print("Saving train summary to ", self.args.model_dir + 'train_summary' + '.json')


    def load_models(self, model_dir, model_epoch='best', load_lm=True):
        print()
        print("[load_models] : Loading from ", model_dir+'model_'+model_epoch )
        self.model.load_state_dict(torch.load(model_dir+'model_'+model_epoch))
        # self.load(dump_pre=model_dir)
        if load_lm:
            print("[load_models] : Loading LM model from  ----------->>>>>>>>>    ", model_dir+'lmmodel_'+model_epoch)
            #self.lm_model.load_state_dict(torch.load('tmp/rhymgan_lm1/model_best')) #model_dir+'lmmodel_'+model_epoch))
            self.lm_model.load_state_dict(torch.load(model_dir+'lmmodel_'+model_epoch))
        self.disc.load_state_dict(torch.load(model_dir+'disc_'+model_epoch))
        self.disc.display_params()


    def _get_disc_word_representation_dictionary(self, vocab):
        all_embs = {}
        for w in vocab: #['head','bread']:#sonnet_vocab:
            if w==utils.UNKNOWN:
                continue
            s1 = self.disc.g_indexer.w_to_idx(w.lower())
            if self.args.use_eow_in_enc:
                assert False
            e1 = self.disc.g2pmodel.encode(s1)
            e1_numpy = e1.data.cpu().numpy().reshape(-1)
            all_embs[w] = e1_numpy
        return all_embs


    def analysis(self, epoch='best', args=None):
        
        model = self.model
        model.eval()
        # self.disc.eval()
        model_dir = self.args.model_dir
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # batch_size=32
        # all_line_endings_train = []
        # for split in ['train','val','test']:
        #     #Compute matrices
        #     num_batches = self.get_num_batches(split, batch_size)
        #     epoch_val_loss = 0.0
        #     ctr = 0
        #     all_loss_tracker_val = {}
        #     dump_info_all = []
        #     for batch in range(num_batches):
        #         len_output = 0
        #         x, _, x_start = self.get_batch(i=batch, batch_size=batch_size, split=split)  # , typ=args.data_type)
        #         for x_i in x:
        #             line_endings_train = []
        #             for w in x_i[:-1]: # just removing end?
        #                 word_idx = self.g_indexer.idx2w[w] 
        #                 line_endings_train.append(word_idx)
        #             all_line_endings_train.append(line_endings_train)
        # if self.args.data_type=="limerick":       
        #     json.dump(all_line_endings_train, open('all_line_endings_limerick.json', 'w'))
        # elif self.args.data_type=="sonnet_endings":   
        #     json.dump(all_line_endings_train, open('all_line_endings_sonnet.json', 'w'))
        # 0/0

        # Analyzing learnt representations
        if self.args.data_type == LIMERICK_DATASET_IDENTIFIER:
            sonnet_vocab = self.limerick_vocab
        else:
            sonnet_vocab = self.sonnet_vocab
        all_embs = {}
        dump_file = model_dir + "all_embs_epoch"+epoch+".pkl"
        print("dumping to ",dump_file)
        sonnet_vocab[0:5], len(sonnet_vocab)
        self.model.eval()
        # for w in sonnet_vocab: #['head','bread']:#sonnet_vocab:
        #     if w==utils.UNKNOWN:
        #         continue
        #     s1 = self.disc.g_indexer.w_to_idx(w.lower())
        #     # print(w,s1)
        #     if self.args.use_eow_in_enc:
        #         assert False
        #     e1 = self.disc.g2pmodel.encode(s1)
        #     e1_numpy = e1.data.cpu().numpy().reshape(-1)
        #     all_embs[w] = e1_numpy
        #     #break
        all_embs = self._get_disc_word_representation_dictionary(sonnet_vocab)
        pickle.dump(all_embs, open(dump_file,'wb'))
        print("DUMPING TO ", dump_file)
        # Analyzing dot products can be done separately
        
        #get_spelling_baseline_for_rhyming = True #False
        #eval_info = self.rhyming_eval.analyze_embeddings_for_rhyming_from_dict(all_embs, get_spelling_baseline_for_rhyming, spelling_type='last1')
        eval_info = self.rhyming_eval.analyze_embeddings_for_rhyming_from_dict(all_embs)
        thresh_f1, maxf1, test_f1 = eval_info['thresh_f1'], eval_info['maxf1'], eval_info['test_f1']
        print("[EPOCH] = ", epoch, " ---->> thresh_f1, maxf1, test_f1 = ", thresh_f1, maxf1, test_f1)


        ##### GET SAMPLES
        print("=======GET SAMPLES======")
        model.eval()
        samples = []
        x, _, x_start = self.get_batch(i=0, batch_size=331, split='val')  # , typ=args.data_type)
        for i in range(10000):
            try:
                with time_limit(4):
                    info = model(gt=None, use_gt=False, temperature=args.temperature)
                    sample_str = self.g_indexer.idx_to_2([i.data.cpu().item() for i in info['actions']])
                    samples.append(sample_str)
            except TimeoutException as e:
                print("Timed out!")
            if i%500==0:
                print("Done with ", i+1, " samples")
        if not os.path.exists(self.args.model_dir + 'samples_analysis/'):
            os.makedirs(self.args.model_dir + 'samples_analysis/')
        json.dump(samples, open(self.args.model_dir + 'samples_analysis/' + str(epoch) + '.json', 'w'))
        print("="*99)
        print("DONE WITH SAMPLES")

        pattern_eval_info = self.rhyming_eval.analyze_samples_from_endings(samples)
        print(json.dumps(pattern_eval_info, indent=4))


        ##### PPL AND INFO
        batch_size = 32
        for split in ['val','test']:

            #Computing ppl of model
            num_batches = self.get_num_batches(split, batch_size)
            epoch_val_loss = 0.0
            ctr = 0
            all_loss_tracker_val = {}
            for batch in range(num_batches):
                total_batch_loss, len_output, info = self.get_loss(split, batch, batch_size, mode='eval',
                                                                   criterion=criterion)
                epoch_val_loss = epoch_val_loss + total_batch_loss.data.cpu().item()
                for k, v in info.items():
                    if k.count('loss') > 0:
                        if k not in all_loss_tracker_val:
                            all_loss_tracker_val[k] = 0.0
                        all_loss_tracker_val[k] += v
                ctr = ctr + len_output
                #break
            print("[ENDINGS MODEL] epoch =" , epoch, "split = ", split, "  epoch_val_loss  = ", epoch_val_loss, " ctr = ", ctr, \
                "PPL = ", np.exp(epoch_val_loss/ctr))
            print("="*99)
            print("DONE WITH PPL")

            #Compute matrices
            if args.dump_matrices:
                limit = 5
                num_batches = min(limit,self.get_num_batches(split, batch_size))
                epoch_val_loss = 0.0
                ctr = 0
                all_loss_tracker_val = {}
                dump_info_all = []
                for batch in range(num_batches):
                    len_output = 0
                    x, _, x_start = self.get_batch(i=batch, batch_size=batch_size, split=split)  # , typ=args.data_type)
                    all_e = []
                    for x_i in x:
                        len_output += len(x_i)
                        try:
                            with time_limit(10):
                                # long_function_call()
                                info = model(gt=None, use_gt=False)  # Get Sample
                                line_endings_gen = info['actions']  # [i.data.cpu().item() for i in info['actions']] #info['actions']
                                line_endings_train = []
                                # print("info = ", info)
                                for w in x_i[:-1]: # just removing end?
                                    word_idx = torch.tensor(w)
                                    if self.args.use_cuda:
                                        word_idx = word_idx.cuda()
                                    line_endings_train.append(word_idx)
                                disc_info = self.disc.update_discriminator(line_endings_gen, line_endings_train, True, self.g_indexer.idx2w)
                                dump_info = disc_info['dump_info']
                                dump_info.update({'line_endings_gen':[self.g_indexer.idx2w[idx.data.cpu().item()] for idx in line_endings_gen], \
                                    'line_endings_train':[self.g_indexer.idx2w[idx.data.cpu().item()] for idx in line_endings_train]})
                                dump_info_all.append(dump_info)
                        except TimeoutException as e:
                            print("Timed out!")
                print("="*99)
                print("DONE WITH MATRICES")

            if not os.path.exists(self.args.model_dir + 'dump_info_all_analysis/'):
                os.makedirs(self.args.model_dir + 'dump_info_all_analysis/')
            pickle.dump(dump_info_all, open(self.args.model_dir + 'dump_info_all_analysis/dump_info_all' + str(split) + '_epoch'+epoch+'.pkl', 'wb'))
            
            print("="*99)
            print("DONE WITH DUMPING MATRICES")

    def lm_analysis(self, epoch='best', args=None):

        model = self.lm_model
        model.eval()
        model_dir = self.args.model_dir
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        ##### PPL AND INFO
        batch_size = 32
        for split in ['val', 'test']:

            print("Split: ", split)
            # Computing ppl of model
            num_batches = self.get_num_batches(split, batch_size)
            epoch_val_loss = 0.0
            ctr = 0
            all_loss_tracker_val = {}
            for batch in range(num_batches):
                total_batch_loss, len_output, info = self.get_loss(split, batch, batch_size, mode='eval',
                                                                   criterion=criterion)
                epoch_val_loss = epoch_val_loss + total_batch_loss.data.cpu().item()
                for k, v in info.items():
                    if k.count('loss') > 0:
                        if k not in all_loss_tracker_val:
                            all_loss_tracker_val[k] = 0.0
                        all_loss_tracker_val[k] += v
                ctr = ctr + len_output
                # break
            print("[ENDINGS MODEL] epoch =", epoch, "split = ", split, "  epoch_val_loss  = ", epoch_val_loss,
                  " ctr = ", ctr, \
                  "PPL = ", np.exp(epoch_val_loss / ctr))


            with torch.no_grad():
                # We have a validation/test set, so compute all the metrics on it.
                # Note that if we set hier_mode=False, code will compute ppl for vanilla-LM
                val_loss, num_batches, n_words, n_samples = self.validation_loss_lm(split=split, batch_size=64, hier_mode=True)
                # val_metrics = get_metrics(val_loss, num_batches)

                print("Total val_loss : ", val_loss, " and n_words : ", n_words)
                print("Ppl: ", np.exp((epoch_val_loss + val_loss)/(ctr + n_words)))
            print("DONE WITH PPL")
            print("-" * 50)







