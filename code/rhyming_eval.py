import json
import numpy as np
import utils
import pickle

tmp_dir = 'tmp/'
model_name = 'lstm_sonnet_reinforce0_1_allsonnet_ae'
epoch = '40'

from constants import *

cmu_dict = utils.loadCMUDict('data/resources/cmudict-0.7b.txt')

class RhymingEval:
    
    def __init__(self, dataset_identifier:str = SONNET_ENDINGS_DATASET_IDENTIFIER):
        self.dataset_identifier = dataset_identifier
        assert dataset_identifier in [SONNET_ENDINGS_DATASET_IDENTIFIER, LIMERICK_DATASET_IDENTIFIER]
        if self.dataset_identifier == SONNET_ENDINGS_DATASET_IDENTIFIER:
            self.interesting_patterns = ['0011','0110','0101']
        else:
            self.interesting_patterns = ['00110']
        
    def __str__(self):
        return 'RhymingEval[dataset_identifier={},interesting_patterns={}]'\
              .format(self.dataset_identifier,self.interesting_patterns)

    def _group_by_rhyming(self):
        ret = {}
        for w,p in cmu_dict.items():
            pr = utils.get_rhyming_part(p)
            #print(p,pr)
            if pr not in ret:
                ret[pr] = []
            ret[pr].append(w)
            #break
        return ret

    def setup(self, line_endings_file_location, line_endings_file_location_test):
        # rhy_to_words = self._group_by_rhyming()
        # print(len(rhy_to_words), list(rhy_to_words.items())[10])
        # rhy_to_words_items = sorted(list(rhy_to_words.items()), key=lambda x: -len(x[1]))
        all_sonnet_line_endings = json.load(open(line_endings_file_location,'r'))
        all_sonnet_line_endings_test = json.load(open(line_endings_file_location_test,'r'))
        # if self.dataset_identifier == SONNET_ENDINGS_DATASET_IDENTIFIER:
        #     all_sonnet_line_endings = json.load(open('all_line_endings_sonnet.json','r'))
        # else:
        #     all_sonnet_line_endings = json.load(open('all_line_endings_limerick.json','r'))            
        #6*len(all_sonnet_line_endings)
        #TODO: do split-wise. load different splits

        def process_endings(data):
            rhyming_pairs = []
            non_rhyming = []
            pattern_cnt = {}
            for line in data:
                pattern, failure = utils.compute_rhyming_pattern(line, cmu_dict)
                if not failure:
                    pattern = ''.join([str(p) for p in pattern])
                    if pattern not in pattern_cnt:
                        pattern_cnt[pattern] = 0
                    pattern_cnt[pattern] += 1
                for i in range(len(line)):
                    for j in range(i+1,len(line)):
                        w1 = line[i]
                        w2=line[j]
                        if w1 in cmu_dict and w2 in cmu_dict:
                            p1 = utils.get_rhyming_part(cmu_dict[w1])
                            p2 = utils.get_rhyming_part(cmu_dict[w2])
                            if p1==p2:
                                rhyming_pairs.append([w1,w2])
                            else:
                                non_rhyming.append([w1,w2])
            return rhyming_pairs, non_rhyming, pattern_cnt

        rhyming_pairs, non_rhyming, pattern_cnt = process_endings(all_sonnet_line_endings)
        self.pattern_cnt_sum = sum(pattern_cnt.values())
        self.rhyming_pairs, self.non_rhyming, self.pattern_cnt = rhyming_pairs, non_rhyming, pattern_cnt
        print("------>>> rhyming_eval: len(self.rhyming_pairs) = ", len(self.rhyming_pairs),
              "\n -- len(self.non_rhyming) = ", len(self.non_rhyming) )

        rhyming_pairs_test, non_rhyming_test, pattern_cnt_test = process_endings(all_sonnet_line_endings_test)
        self.pattern_cnt_sum_test = sum(pattern_cnt_test.values())
        self.rhyming_pairs_test, self.non_rhyming_test, self.pattern_cnt_test = rhyming_pairs_test, non_rhyming_test, pattern_cnt_test
        print("------>>> rhyming_eval: len(self.rhyming_pairs_test) = ", len(self.rhyming_pairs_test),
              "\n -- len(self.non_rhyming_test) = ", len(self.non_rhyming_test))


    def _get_spelling_baseline(self, w1, w2, spelling_type='last3'):
        if spelling_type=='last3':
            return w1[-3:] == w2[-3:]
        elif spelling_type=='last2':
            return w1[-2:] == w2[-2:]
        elif spelling_type=='last4':
            return w1[-4:] == w2[-4:]
        elif spelling_type=='last1':
            return w1[-1:] == w2[-1:]
        else:
            assert False


    ###### emb
    def _load_cosines(self, emb, get_spelling_baseline_for_rhyming, spelling_type, split='dev'):
        if split == 'test':
            rhyming_pairs = self.rhyming_pairs_test
            non_rhyming = self.non_rhyming_test
        else:
            rhyming_pairs = self.rhyming_pairs
            non_rhyming = self.non_rhyming
        all_cosines_r = []
        all_cosines_nr = []
        for r in rhyming_pairs:
            w1,w2 = r
            if w1 in emb and w2 in emb:
                if get_spelling_baseline_for_rhyming:
                    if self._get_spelling_baseline(w1,w2, spelling_type):
                        all_cosines_r.append(1)
                    else:
                        all_cosines_r.append(0)
                else:
                    e1_numpy, e2_numpy = emb[w1], emb[w2]
                    all_cosines_r.append(np.sum(e1_numpy * e2_numpy)/ np.sqrt( (np.sum(e1_numpy * e1_numpy) * np.sum(e2_numpy * e2_numpy))))
        for r in non_rhyming:
            w1,w2 = r
            if w1 in emb and w2 in emb:
                if get_spelling_baseline_for_rhyming:
                    if self._get_spelling_baseline(w1,w2, spelling_type):
                        all_cosines_nr.append(1)
                    else:
                        all_cosines_nr.append(0)
                else:
                    e1_numpy, e2_numpy = emb[w1], emb[w2]
                    all_cosines_nr.append(np.sum(e1_numpy * e2_numpy)/ np.sqrt( (np.sum(e1_numpy * e1_numpy) * np.sum(e2_numpy * e2_numpy))))
        return np.array(all_cosines_r), np.array(all_cosines_nr)
     
        
    def analyze_embeddings_for_rhyming(self, emb_loc, get_spelling_baseline_for_rhyming=False, spelling_type=None):
        emb = pickle.load(open(emb_loc,'rb'))
        return self.analyze_embeddings_for_rhyming_from_dict(emb)
        
        
    def analyze_embeddings_for_rhyming_from_dict(self, emb, get_spelling_baseline_for_rhyming=False, spelling_type=None):

        all_cosines_r,all_cosines_nr = self._load_cosines(emb, get_spelling_baseline_for_rhyming, spelling_type)
        maxf1 = 0
        thresh_f1 = -1
        thresh=0.5
        # details = []
        while thresh<0.95: #[0.8,0.75, 0.77, 0.73, 0.70, 0.65, 0.55, 0.60, 0.63, 0.85, 0.86]:
            vals = [sum(all_cosines_r>thresh), sum(all_cosines_nr>thresh), sum(all_cosines_r<=thresh), sum(all_cosines_nr<=thresh)]
            prec = vals[0]/(vals[0]+vals[1])
            rec = vals[0]/(vals[0]+vals[2])
            f1 = 2*prec*rec/(prec+rec)
            if f1>maxf1:
                maxf1 = f1
                thresh_f1 = thresh
            print("thresh, prec, rec, f1 = ", thresh, prec, rec, f1)
            thresh+=0.01 #0.001
        print("=========  len(all_cosines_r), len(all_cosines_nr), thresh_f1, maxf1 =",  len(all_cosines_r), len(all_cosines_nr), thresh_f1, maxf1)

        all_cosines_r, all_cosines_nr = self._load_cosines(emb, get_spelling_baseline_for_rhyming, spelling_type, split='test')
        thresh = thresh_f1
        vals = [sum(all_cosines_r > thresh), sum(all_cosines_nr > thresh), sum(all_cosines_r <= thresh),
                sum(all_cosines_nr <= thresh)]
        prec = vals[0] / (vals[0] + vals[1])
        rec = vals[0] / (vals[0] + vals[2])
        maxf1_test = 2 * prec * rec / (prec + rec)
        print(" -test- len(all_cosines_r), len(all_cosines_nr), thresh, prec, rec, f1 = ", len(all_cosines_r), len(all_cosines_nr), thresh, prec, rec, maxf1_test)        
        #print(" -test- thresh, prec, rec, f1 = ", thresh, prec, rec, maxf1_test)
        print("=========  maxf1_test =", maxf1_test)
        return {'thresh_f1':thresh_f1, 'maxf1':maxf1,'test_f1':maxf1_test}
        

    ## pattern eval
    def analyze_samples_from_endings(self, samples):
        from collections import defaultdict
        epoch_pattern_cnt = defaultdict(lambda: 0)
        total_count = 0
        for sample in samples:
            pattern, failure = utils.compute_rhyming_pattern(sample, cmu_dict)
            if not failure:
                pattern = ''.join([str(p) for p in pattern])
                if pattern not in epoch_pattern_cnt:
                    epoch_pattern_cnt[pattern] = 0
                epoch_pattern_cnt[pattern] += 1
                total_count += 1
        print("[ANALYZING SAMPLES] [model patterns] epoch_pattern_count of samples = ", \
              json.dumps(epoch_pattern_cnt, indent=4))
        pattern_dist = {k:v*1.0/total_count for k,v in sorted(epoch_pattern_cnt.items(), key=lambda x:-x[1])}
        print("[ANALYZING SAMPLES] [modeel_patterns] epoch_pattern_count of samples (in %) = ", \
              json.dumps(pattern_dist,indent=4))
        data_pattern_dist = {k:v*1.0/self.pattern_cnt_sum for k,v in sorted(self.pattern_cnt.items(), key=lambda x:-x[1])}
        print("[ANALYZING SAMPLES] [dataset patterns] epoch_pattern_count of samples (in %) = ", \
              json.dumps(data_pattern_dist,indent=4))
        print()
        f = sum([epoch_pattern_cnt[pattern] for pattern in self.interesting_patterns])*1.0/total_count
        summary = {}
        summary['pattern_dist_model'] = pattern_dist
        summary['pattern_dist_data'] = data_pattern_dist
        #summary['kl'] = kl
        summary['pattern_success_ratio'] = f
        summary['pattern_sampling_rate'] = 1.0/f if f!=0 else float("inf")
        summary['pattern_cnt_of_samples_tested'] = total_count
        summary['patterns_tested_for_success'] = self.interesting_patterns
        return summary
    
    '''
    Assumes a specific format.
    Samples is a list of strings, each string represents one poem
    '''
    def analyze_samples_from_poetry_samples(self, samples):
        endings = []
        for sample in data:
            sample = sample.strip().split(' <eos>')[:-1]
            sample = [line.strip().split()[-1] for line in sample]
            endings.append(sample)
        return analyze_samples_from_endings(endings)
        
                
