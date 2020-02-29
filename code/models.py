from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from utils import Indexer
import utils
from constants import *


################
class Model(nn.Module):

    def __init__(self, H, i_size=None, o_size=None, emsize=128, start_idx=1, end_idx=2, typ='encdec', args=None,
                 tie_inp_out_emb=False):
        super(Model, self).__init__()
        self.H = H
        self.emsize = emsize
        self.emb_i = nn.Embedding(i_size, emsize)
        if tie_inp_out_emb:
            self.emb_o = self.emb_i
        else:
            self.emb_o = nn.Embedding(o_size, emsize)
        self.encoder = nn.LSTMCell(emsize, H)
        self.decoder = nn.LSTMCell(emsize, H)
        self.softmax = nn.Softmax()
        self.decoder_layer = nn.Linear(H, o_size)
        self.sig = nn.Sigmoid()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.typ = typ
        self.use_cuda = args.use_cuda

    def encode(self, e1):
        h = torch.zeros(1, self.H), torch.zeros(1, self.H)
        if self.use_cuda:
            h = h[0].cuda(), h[1].cuda()
        for ch in e1:
            ch_idx = torch.tensor(ch)
            if self.use_cuda:
                ch_idx = ch_idx.cuda()
            ch_emb = self.emb_i(ch_idx).view(1, -1)
            h = self.encoder(ch_emb, h)
        out, c = h
        return c

    def decode(self, enc, gt, use_gt=True, max_steps=11):
        info = {}
        h = enc, enc
        sz = len(gt)
        if not use_gt:
            sz = max_steps
        ch = self.start_idx
        ch_idx = ch
        out_all = []
        prediction = []
        for i in range(sz):
            ch_idx = torch.tensor(ch_idx)
            if self.use_cuda:
                ch_idx = ch_idx.cuda()
            ch_emb = self.emb_o(ch_idx).view(1, -1)
            h = self.decoder(ch_emb, h)
            out, _ = h
            out = self.decoder_layer(out)
            out_all.append(out)
            if not use_gt:
                preds = self.softmax(out)
                pred = torch.argmax(preds)
                if pred.cpu().numpy() == self.end_idx:
                    break
                ch_idx = pred
                prediction.append(ch_idx.data.cpu().item())
            else:
                ch_idx = gt[i]
        if use_gt:
            return out_all, info
        else:
            return prediction, info


################
class GeneratorModel(nn.Module):

    def __init__(self, H, o_size=None, emsize=128, start_idx=1, end_idx=2, typ='dec', args=None, use_gan=False,
                 unk_idx=None):
        super(GeneratorModel, self).__init__()
        self.H = H
        self.emsize = emsize
        self.emb_o = nn.Embedding(o_size, emsize)
        self.phonetic_emb = nn.Embedding(o_size, emsize)
        self.decoder = nn.LSTMCell(emsize, H)
        self.softmax = nn.Softmax()
        self.decoder_layer = nn.Linear(H, o_size)
        self.sig = nn.Sigmoid()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.typ = typ
        self.use_cuda = args.use_cuda
        self.use_gan = use_gan
        assert not use_gan
        self.emsize = emsize
        self.args = args
        self.unk_idx = unk_idx

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if self.use_cuda:
            eps = eps.cuda()
        return eps.mul(std).add_(mu)

    def freeze_emb(self):
        self.emb_o.weight.requires_grad = False
        print("[GeneratorModel] -- FREEZING word embedding emb_o")

    def display_params(self):
        print("[GeneratorModel]: model parametrs")
        for name, param in self.named_parameters():
            print("name=", name, " || grad:", param.requires_grad, "| size = ", param.size())

    def load_gutenberg_we(self, dict_pickle_pth, indexer_idx2w):
        sz = len(indexer_idx2w)
        emb = np.random.rand(sz, self.emsize)
        print(emb.shape)
        pretrained = {}
        lines = open(dict_pickle_pth, 'r').readlines()
        for line in lines:
            line = line.strip().split()
            w = line[0]
            e = np.array([float(val) for val in line[1:]])
            pretrained[w] = e
        found = 0
        for i in range(sz):
            word = indexer_idx2w[i]
            if word in pretrained:
                emb[i, :] = pretrained[word]
                found += 1
        print("load_gutenberg_we: found", found, " out of ", sz)
        self.emb_o.weight.data.copy_(torch.from_numpy(emb))

    def forward(self, gt=None, use_gt=True, max_steps=4, temperature=1.0):
        if self.use_gan:
            enc = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, self.H)))
        else:
            enc = torch.zeros(1, self.H)
        if self.use_cuda:
            enc = enc.cuda()
        h = enc, enc  # torch.zeros(1,self.H), torch.zeros(1,self.H)
        if not use_gt:
            if self.args.data_type == "sonnet_endings":
                sz = max_steps = 4
            elif self.args.data_type == "limerick":
                sz = max_steps = 5
            else:
                assert False
        else:
            sz = max_steps = len(gt)
        word = self.start_idx
        word_idx = word
        out_all = []
        prediction = []
        batch_size = 1
        probs, states, actions = [], [], []
        for i in range(sz):
            word_idx = torch.tensor(word_idx)
            if self.use_cuda:
                word_idx = word_idx.cuda()
            # print(i,ch_idx)
            word_emb = self.emb_o(word_idx).view(1, -1)
            h = self.decoder(word_emb, h)
            out, _ = h
            out = self.decoder_layer(out)
            out = out / temperature
            out_all.append(out)
            action_probs = self.softmax(out)
            if not use_gt:
                torch_distribution = Categorical(action_probs.view(batch_size, -1))
                not_done = True
                attempts = 21
                while not_done:
                    attempts -= 1
                    action = torch_distribution.sample()
                    log_prob_action = torch_distribution.log_prob(action)
                    action_idx = action.data[0]
                    if action_idx.cpu().item() == self.end_idx:
                        continue
                    if action_idx.cpu().item() == self.unk_idx:
                        continue
                    word_idx = action_idx
                    probs.append(log_prob_action)
                    states.append(h)
                    actions.append(action_idx)
                    not_done = False
            else:
                word_idx = gt[i]
        return {'out_all': out_all, 'logprobs': probs, 'states': states, 'actions': actions}



####################


class CNN(nn.Module):
    def __init__(self, args, reduced_size=None, info={}):
        super(CNN, self).__init__()
        # disc_type=DISC_TYPE_MATRIX
        self.disc_type = disc_type = args.disc_type
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, padding=0),
            nn.ReLU())
        # 1,4,3,3
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=2),
            nn.ReLU())
        # 1,8,2,2
        ## but for 5 lines, it is 1,8,3,3
        if args.data_type == "sonnet_endings":
            self.scorer = nn.Linear(2 * 2 * 8, 1)
        elif args.data_type == "limerick":
            self.scorer = nn.Linear(3 * 3 * 8, 1)
        self.predictor = nn.Sigmoid()
        self.args = args
        self.use_cuda = args.use_cuda

        ##
        self.g_indexer = Indexer(args)
        self.g_indexer.load('tmp/tmp_' + args.g2p_model_name + '/solver_g_indexer')
        self.g2pmodel = Model(H=info['H'], args=args, i_size=self.g_indexer.w_cnt, o_size=self.g_indexer.w_cnt,
                              start_idx=self.g_indexer.w2idx[utils.START])
        if not args.learn_g2p_encoder_from_scratch:
            print("=====" * 7, "LOADING g2p ENCODER PRETRAINED")
            model_dir = 'tmp/tmp_' + args.g2p_model_name + '/'
            state_dict_best = torch.load(model_dir + 'model_best')
            self.g2pmodel.load_state_dict(state_dict_best)
        if not args.trainable_g2p:
            assert not args.learn_g2p_encoder_from_scratch
            for param in self.g2pmodel.parameters():
                param.requires_grad = False

    def display_params(self):
        print("=" * 44)
        print("[CNN]: model parametrs")
        for name, param in self.named_parameters():
            print("name=", name, " || grad:", param.requires_grad, "| size = ", param.size())
        print("=" * 44)

    def _compute_word_reps(self, words_str, deb=False):
        if deb:
            print("words_str = ", words_str)
        use_eow_marker = self.args.use_eow_in_enc
        assert not use_eow_marker, "Not yet tested"
        word_reps = [self.g_indexer.w_to_idx(s1) for s1 in words_str]
        if self.args.use_eow_in_enc:
            x_end = self.g_indexer.w2idx[utils.END]
            word_reps = [x_i + [x_end] for x_i in word_reps]
        word_reps = [self.g2pmodel.encode(w) for w in word_reps]
        return word_reps

    def _compute_pairwise_dot(self, measure_encodings_b):
        ret = []
        sz = len(measure_encodings_b)
        for measure_encodings_b_t in measure_encodings_b:
            for measure_encodings_b_t2 in measure_encodings_b:
                t1 = torch.sum(measure_encodings_b_t * measure_encodings_b_t2)
                t2 = torch.sqrt(torch.sum(measure_encodings_b_t * measure_encodings_b_t))
                t3 = torch.sqrt(torch.sum(measure_encodings_b_t2 * measure_encodings_b_t2))
                assert t2 > 0
                assert t3 > 0, "t3=" + str(t3)
                ret.append(t1 / (t2 * t3))
        ret = torch.stack(ret)
        ret = ret.view(sz, sz)
        return ret

    def _score_matrix(self, x, deb=False):
        x = x[0].unsqueeze(0).unsqueeze(0)  # -> 1,1,ms,ms
        if deb:
            print("---x.shape = ", x.size())
        out = self.layer1(x)
        if deb:
            print("---out = ", out.size(), out)
        out = self.layer2(out)
        if deb:
            print("---out = ", out.size(), out)
        out = out.view(out.size(0), -1)  # arrange by bsz
        score = self.scorer(out)
        if deb:
            print("---out sum = ", torch.sum(out))
            print("---score = ", score)
        prob = self.predictor(score)
        return {'prob': prob, 'out': out, 'score': score}

    def _compute_rhyming_matrix(self, words_str, deb=False):
        word_reps = self._compute_word_reps(words_str)
        rhyming_matrix = self._compute_pairwise_dot(word_reps)
        return rhyming_matrix, words_str

    def _compute_rnn_on_word_reps(self, word_reps):
        h = torch.zeros(1, self.linear_rep_H), torch.zeros(1, self.linear_rep_H)
        if self.use_cuda:
            h = h[0].cuda(), h[1].cuda()
        for w in word_reps:
            h = self.linear_rep_encoder(w, h)
        out, c = h
        return c

    def _run_discriminator(self, words_str, deb):
        rhyming_matrix, words_str = self._compute_rhyming_matrix(words_str, deb)
        vals = self._score_matrix([rhyming_matrix])
        vals.update({'rhyming_matrix': rhyming_matrix, 'linear_rep': None, 'words_str': words_str})
        return vals

    def update_discriminator(self, line_endings_gen, line_endings_train, deb=False, word_idx_to_str_dict=None):
        eps = 0.0000000001
        ret = {}
        dump_info = {}
        words_str_train = [word_idx_to_str_dict[word_idx.data.cpu().item()] for word_idx in line_endings_train]
        words_str_gen = [word_idx_to_str_dict[word_idx.data.cpu().item()] for word_idx in line_endings_gen]
        disc_real = self._run_discriminator(words_str_train, deb)
        if deb:
            print("rhyming_matrix_trai = ", disc_real['rhyming_matrix'], "|| prob = ", disc_real['prob'])
            if self.args.disc_type == DISC_TYPE_MATRIX:
                dump_info['rhyming_matrix_trai'] = disc_real['rhyming_matrix'].data.cpu().numpy()
            dump_info['real_prob'] = disc_real['prob'].data.cpu().item()
            dump_info['real_words_str'] = disc_real['words_str']
        disc_gen = self._run_discriminator(words_str_gen, deb)
        if deb:
            print("rhyming_matrix_gen = ", disc_gen['rhyming_matrix'], "|| prob = ", disc_gen['prob'])
            if self.args.disc_type == DISC_TYPE_MATRIX:
                dump_info['rhyming_matrix_gen'] = disc_gen['rhyming_matrix'].data.cpu().numpy()
            dump_info['gen_prob'] = disc_gen['prob'].data.cpu().item()
            dump_info['gen_words_str'] = disc_gen['words_str']
        prob_real = disc_real['prob']
        prob_gen = disc_gen['prob']
        loss = -torch.log(prob_real + eps) - torch.log(1.0 - prob_gen + eps)
        reward = prob_gen
        if self.args.use_score_as_reward:
            reward = disc_gen['score']
        ret.update({'loss': loss, 'reward': reward, 'dump_info': dump_info})
        return ret




