import numpy as np
import os
import codecs
import torch
import pickle


####
UNKNOWN='@@UNKNOWN@@'
START='@start@'
END='@end@'
PAD='@pad@'
EOW='@eow@'

###################
def load_gutenberg_word2vec(pth):
    data = [r.strip().split() for r in open(pth,"r").readlines()]
    word_2_emb = {r[0]:np.array([float(v) for v in r[1:]]) for r in data}
    return word_2_emb

def load_sonnet_vocab(pth):
    data = [r.strip() for r in open(pth,"r").readlines()]
    return data


#########################
# Modified indexer
class Indexer:

    def __init__(self, args):
        # self.w2idx = {'start': 1, 'pad': 0, 'end': 2}
        self.w2idx = {START: 1, PAD: 0, END: 2, UNKNOWN:3}
        # self.w2idx = {'pad': 0}
        if args.data_type == "sonnet_endings":
            self.w2idx[EOW] = 4
        self.w_cnt = len(self.w2idx)
        self.idx2w = None

    def process(self, lst_of_lst):
        for item in lst_of_lst:
            for v in item:
                if v not in self.w2idx:
                    self.w2idx[v] = self.w_cnt
                    self.w_cnt += 1
        self.idx2w = {i: w for w, i in self.w2idx.items()}

    def w_to_idx(self, w, is_char_level=False):
        # print("w=",w)
        if UNKNOWN not in self.w2idx:
            self.w2idx[UNKNOWN] = self.w2idx[PAD] #TODO: Need a permanent fix for this
        if is_char_level:
            w = [str(ch) for ch in w if ord(ch) < 128]
        return [self.w2idx[ch] if ch in self.w2idx else self.w2idx[UNKNOWN] for ch in w]

    def idx_to_2(self, w):
        return [self.idx2w[ch] for ch in w]

    def load(self, prefix):
        self.w2idx = pickle.load( open(prefix+'.w2idx.pkl','rb'))
        self.idx2w = pickle.load( open(prefix+'.idx2w.pkl','rb'))
        self.w_cnt = pickle.load( open(prefix+'.w_cnt.pkl','rb'))

    def save(self, prefix):
        pickle.dump(self.w2idx, open(prefix+'.w2idx.pkl','wb'))
        pickle.dump(self.idx2w, open(prefix+'.idx2w.pkl','wb'))
        pickle.dump(self.w_cnt, open(prefix+'.w_cnt.pkl','wb'))


def get_char_seq_from_word_seq(self, lst_of_words, w2idx, use_eow_marker, eow_marker=None):
    #lst of words
    #each word is list of chars
    #consider end of word_sep
    # retuerns a lit of units for indexing
    ret = []
    for j,w in enumerate(lst_of_words):
        ret.extend(w)
        if use_eow_marker and j<(len(lst_of_words)-1):
            ret.append(eow_marker)
    # ret_idx = [self.g_indexer.w2idx[ch] for ch in ret]
    ret_idx = [w2idx[ch] for ch in ret]
    return {'x':ret, 'indexed_x':ret_idx}


################


def preproLine(line, lower=True, ascii_only=True, remove_punc=False):
    if lower:
        line = line.lower()
    if ascii_only:
        line = ''.join([str(ch) for ch in line if ord(ch)<=127])
    if remove_punc:
        line = ''.join([str(ch) for ch in line if (ch>='A' and ch<='Z') or (ch>='a' and ch<='z') or (ch==' ')])
    return line


def loadCMUDict(fname="../data/cmudict-0.7b.txt"):
    #d = open(fname,"r").readlines() 
    d = []
    with codecs.open(fname, "r",encoding='utf-8', errors='ignore') as fdata:
        for line in fdata:
            d.append(line)
    d = [line for line in d if line[0].isalpha()]
    d = [line.strip().split('  ') for line in d]
    #print d[0]
    ret = { val[0].strip().lower():val[1].strip().split(' ') for val in d }
    return ret



#following function is due to https://github.com/aparrish/rwet-examples/blob/master/pronouncing/cmudict.py
def get_rhyming_part(phones_list):
    """Returns the "rhyming part" of a string with phones. "Rhyming part" here
        means everything from the vowel in the stressed syllable nearest the end
        of the word up to the end of the word."""
    # return get_rhyming_part_deepspeare(phones_list) ### TAKE A NOTE OF THIS
    idx = 0
    for i in reversed(range(0, len(phones_list))):
        if phones_list[i][-1] in ('1', '2'):
            idx = i
            break
    return ' '.join(phones_list[idx:])

def get_rhyming_part_deepspeare(phones_list):
    """Returns the "rhyming part" of a string with phones. "Rhyming part" here
        means everything from the vowel in the syllable nearest the end
        of the word up to the end of the word."""
    idx = 0
    for i in reversed(range(0, len(phones_list))):
        if phones_list[i][-1] in ('1', '2', '0'):
            idx = i
            break
    return ' '.join(phones_list[idx:])



def compute_rhyming_pattern(lst_of_words, cmu_dict):
    rhyming_part_to_idx = {}
    word_to_rhyming_part = []
    not_found = False
    for w in lst_of_words:
        if w not in cmu_dict:
            not_found = True
            return None, not_found
        rhyming_part = get_rhyming_part(cmu_dict[w])
        word_to_rhyming_part.append(rhyming_part)
        if rhyming_part not in rhyming_part_to_idx:
            rhyming_part_to_idx[rhyming_part] = len(rhyming_part_to_idx)
    return [rhyming_part_to_idx[rm] for rm in word_to_rhyming_part], not_found

#########
#Stress
def _extract_stress_pattern_word(word, cmu_dict):
    phones_list = cmu_dict[word]
    ret = []
    for i in range(0, len(phones_list)):
        if phones_list[i][-1] in ('1', '2'):
            ret.append(1)
        elif phones_list[i][-1] in ('0'):
            ret.append(0)
    return ret,phones_list

def _extract_stress_pattern_lst_of_words(lst_of_words, cmu_dict):
    ret = []
    for word in lst_of_words:
        s,p = _extract_stress_pattern_word(word, cmu_dict)
        #print(word,s,p)
        ret.extend(s)
    return ret

def _count_violations(seq):
    ret = 0
    for j in range(len(seq)-1):
        if seq[j]==seq[j+1]:
            ret+=1
    #print("count: seq, ret : ", seq, ret)
    return ret

def count_stress_pattern_violations(lst_of_words, cmu_dict):
    violations = 0
    total_valid = 0
    for i in range(len(lst_of_words)-1):
        w1 = lst_of_words[i]
        w2 = lst_of_words[i+1]  
        if w1 in cmu_dict and w2 in cmu_dict:
            s1,p1 = _extract_stress_pattern_word(w1, cmu_dict)
            #print(w1,s1,p1)
            s2,p2 = _extract_stress_pattern_word(w2, cmu_dict)
            #print(w2,s2,p2)                  
            violations_i = _count_violations(s1+s2)
            #print(violations_i)
            violations+=violations_i
            total_valid+=(len(s1+s2)-1)
            #print()
    return violations, total_valid

##########

def test_rhyming_pattern():
    cmu_data = loadCMUDict()
    print(compute_rhyming_pattern(['read','head','bed'],cmu_data))
    arr = [ ['rose', 'foes', 'descends', 'lends'],
        ['pill', 'will', 'west', 'best'],
        ['round', 'hand', 'sound', 'mused'],
        ['clene', 'hide', 'fears', 'bent'],
        ['poles', 'ground', 'found', 'ground'],
        ['keep', 'hill', 'even', 'knows'],
        ['made', 'knew', 'stage', 'past'],
        ['spend', 'death', 'sea', 'calling'],
        ['fen', 'den', 'artichoke', 'rock'],
        ['wear', 'waste', 'away', 'slay'],
        ['strife', 'life', 'best', 'best'] ]
    for arri in arr:
        print("arr=",arri)
        print("pattern=", compute_rhyming_pattern(arri,cmu_data))

def test_stress_funcs():
    global cmu_dict
    cmu_dict = loadCMUDict()
    print(_extract_stress_pattern_word('bat'))
    print()
    print(_extract_stress_pattern_lst_of_words(['bat','is','near','the','window']))
    print()
    i,o = count_stress_pattern_violations(['bat','is','near','the','window'])
    print("invalid=",i," out of o=",o, " [remaining valid. ignore hwne not foiund in cmu dic]")
    print()
        
# test_rhyming_pattern()
# test_stress_funcs()
