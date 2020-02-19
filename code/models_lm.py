import logging
from overrides import overrides

import numpy as np
import ipdb as pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, LSTM

from typing import Any, Dict, List, Optional, Tuple
from utils import *
from utils_lm import *

logger = logging.getLogger(__name__)
#UNKNOWN = '@@UNKNOWN@@'

class LockedDropout(nn.Module):

    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):

        if not self.training or not dropout:
            return x

        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)

        return mask * x

class VanillaLM(nn.Module):

    def __init__(self,
                 args,
                 vocab_indexer,
                 vocab,
                 decoder_hidden_size = 600,
                 emb_size=128,
                 num_classes=None,
                 start_idx=1,
                 end_idx=2,
                 padding_idx=0,
                 typ='lstm',
                 max_decoding_steps=120,
                 sampling_scheme: str = "first_word",
                 line_separator_symbol: str = "<eos>",
                 reverse_each_line: str = False,
                 n_lines_per_sample: int = 14,
                 tie_weights: bool = True,
                 dropout_ratio: float = 0.3,
                 phoneme_embeddings_dim:int =128,
                 encoder_type: str = None,
                 encoder_input_size: int = 100,
                 encoder_hidden_size: int = 100,
                 encoder_n_layers: int = 1,
                 n_lines_to_gen:int = 4):

        super(VanillaLM, self).__init__()

        self.args = args
        self.vocab_indexer = vocab_indexer
        self.vocab = vocab

        self._scheduled_sampling_ratio = 0.0

        self._max_decoding_steps = max_decoding_steps
        decoder_input_size = emb_size

        self._decoder_input_dim = decoder_input_size
        self._decoder_output_dim = decoder_hidden_size

        self._target_embedder = nn.Embedding(num_classes, emb_size)
        
        self._context_embedder = nn.Embedding(num_classes, phoneme_embeddings_dim) ## TODO: Not clear why this is phoneme_embeddings_dim  

        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.type = typ
        self.use_cuda = args.use_cuda #True

        decoder_embedding_dim = emb_size
        self._target_embedding_dim = decoder_embedding_dim

        assert self.type == "lstm", "Incorrect decoder type"
        self._lm_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._intermediate_projection_layer = Linear(self._decoder_output_dim,
                                                     self._target_embedding_dim)  # , bias=False)
        self._activation = torch.tanh
        self._num_classes = num_classes
        self._output_projection_layer = Linear(self._target_embedding_dim, self._num_classes)

        self._dropout_ratio = dropout_ratio
        self._dropout = nn.Dropout(p=dropout_ratio, inplace=False)
        self._lockdropout = LockedDropout()

        self._encoder_type = encoder_type

        if self._encoder_type is not None:
            self._encoder_input_size = encoder_input_size
            self._encoder_hidden_size = encoder_hidden_size
            self._encoder_namespace = encoder_namespace
            self._encoder = nn.LSTM(input_size=self._encoder_input_size, hidden_size=self._encoder_hidden_size,
                                    batch_first=True, bias=False, num_layers=encoder_n_layers, bidirectional=False)

        if tie_weights:
            # assert self._target_embedding_dim == self._target_embedder.token_embedder_tokens.get_output_dim(), "Dimension mis-match!"
            self._output_projection_layer.weight = self._target_embedder.weight

        # in the config, make these options consistent with those in the reader
        self._sampling_scheme = sampling_scheme  # "first_sentence" # "first_word"
        self.line_separator = line_separator_symbol
        self.reverse_each_line = reverse_each_line
        self.n_lines_per_sample = n_lines_per_sample

        self._n_lines_to_gen = n_lines_to_gen

        self._attention = False
        self.END_SYMBOL = line_separator_symbol

        # self._sonnet_eval = SonnetMeasures()

    def freeze_emb(self, type="_target_embedder"):

        if type == "_target_embedder":
            self._target_embedder.weight.requires_grad = False
        print("[VanillaLM-Model] -- FREEZING target embedder")

    def display_params(self):

        print("[VanillaLM-Model]: model parameters")
        for name, param in self.named_parameters():
            print("name=", name, " || grad:", param.requires_grad)

    def load_gutenberg_we(self, dict_pickle_pth, indexer_idx2w):
        sz = len(indexer_idx2w)
        emb = torch.FloatTensor(sz, self._target_embedding_dim)
        # self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
        torch.nn.init.xavier_uniform_(emb)
        # emb = np.random.rand(sz, self._target_embedding_dim)
        print(emb.shape)

        pretrained = {} #pickle.load(dict_pickle_pth,'rb')
        lines = open(dict_pickle_pth,'r').readlines()
        for line in lines:
            line = line.strip().split()
            w = line[0]
            e = np.array([float(val) for val in line[1:]])
            pretrained[w] = e
        found = 0

        for i in range(sz):
            word = indexer_idx2w[i]
            if word == PAD:
                emb[0].fill_(0)
            elif word in pretrained:
                emb[i,:] = torch.from_numpy(pretrained[word])
                found+=1

        print("load_gutenberg_we: found",found, " out of ", sz)
        # self._target_embedder.weight.data.copy_(torch.from_numpy(emb))
        self._target_embedder.weight.data.copy_(emb)

    def initialize_hidden(self, batch_size, hidden_dim=None, n_dim=2):

        if hidden_dim is None:
            hidden_dim = self._decoder_output_dim

        if n_dim == 2:
            hidden_state = torch.zeros(batch_size, hidden_dim)
            cell_state = torch.zeros(batch_size, hidden_dim)
            if torch.cuda.is_available():
                hidden_state = hidden_state.cuda()
                cell_state = cell_state.cuda()
        else:
            hidden_state = torch.zeros(1, batch_size, hidden_dim)
            cell_state = torch.zeros(1, batch_size, hidden_dim)
            if torch.cuda.is_available():
                hidden_state = hidden_state.cuda()
                cell_state = cell_state.cuda()

        return (hidden_state, cell_state)

    def _sample(self, start_token_idx=None, max_decoding_steps=None,
                ending_words = None, decoder_hidden=None,
                decoder_context=None, conditional=True, debug=False):

        if max_decoding_steps is None:
            max_decoding_steps = self._max_decoding_steps

        if decoder_hidden is None:
            decoder_hidden, decoder_context = self.initialize_hidden(batch_size=1)

        assert not (conditional and ending_words is None), "--exception-- | conditional is set to True and ending_words are not provided!"


        # if conditional:
        #     if not isinstance(ending_words, torch.Tensor):
        #         inp = torch.LongTensor(ending_words)[0]
        #     else:
        #         inp = ending_words[0]
        # else:
        if start_token_idx is None:
            start_token_idx = self.vocab_indexer.w2idx[self.line_separator]

        if not isinstance(start_token_idx, torch.Tensor):
            inp = torch.LongTensor([start_token_idx])  # Token(START_SYMBOL)
        else:
            inp = start_token_idx

        if ending_words is not None:
            n_lines_to_gen = len(ending_words)
            if isinstance(ending_words, list):
                ending_words = torch.LongTensor(ending_words)
            if self.args.use_cuda:
                ending_words = ending_words.cuda()
        else:
            n_lines_to_gen = self._n_lines_to_gen

        if self.args.use_cuda:
            inp = inp.cuda()

        # if self._augment_phoneme_embeddings:
        #     emb_t = torch.cat((self._target_embedder(inp).view(-1), self._context_embedder(inp).view(-1)), dim=-1)  ### this should be target embedded and context embedded
        # else:
        emb_t = self._target_embedder(inp).view(-1)  ### this should be target embedded

        logprobs = []
        actions = []
        actions_idx = []

        logprobs_line = []
        actions_line = []
        actions_idx_line = []

        i2v = self.vocab

        count_lines_gen = 0

        prev_action = self.line_separator

        for t in range(max_decoding_steps):

            decoder_hidden, decoder_context = self._lm_cell(emb_t.unsqueeze(0), (decoder_hidden, decoder_context))

            # output = self._lockdropout(x=decoder_hidden.unsqueeze(1), dropout=self._dropout_ratio)
            output = decoder_hidden.unsqueeze(1)

            pre_decoded_output = self._intermediate_projection_layer(output.view(1, -1))
            decoded_output = self._output_projection_layer(pre_decoded_output)
            logits = decoded_output.view(1, decoded_output.size(1))

            logprobs_line.append(F.log_softmax(logits, dim=-1))
            class_probabilities = F.softmax(logits, dim=-1)

            predicted_action = UNKNOWN
            while predicted_action == UNKNOWN:
                predicted_action_idx = torch.multinomial(class_probabilities, 1)
                predicted_action = i2v[predicted_action_idx.data.item()]

            if prev_action == self.line_separator and conditional:
                predicted_action_idx = ending_words[count_lines_gen]
                predicted_action = i2v[ending_words[count_lines_gen].data.item()]
                prev_action = predicted_action
            else:
                predicted_action_idx = predicted_action_idx[0]
                prev_action = predicted_action

            actions_line.append(predicted_action)
            actions_idx_line.append(predicted_action_idx)

            inp = predicted_action_idx

            # if self._augment_phoneme_embeddings:
            #     emb_t = torch.cat((self._target_embedder(inp).view(-1), self._context_embedder(inp).view(-1)), dim=-1)
            # else:
            emb_t = self._target_embedder(inp).view(-1)

            # all_predictions_indices.append(last_predictions)
            # last_predictions_str = i2v[last_predictions.data.item()]

            if predicted_action == self.line_separator:
                actions.append(actions_line[:-1])
                logprobs.append(logprobs_line[:-1])
                actions_idx.append(actions_idx_line[:-1])

                actions_line = []
                logprobs_line = []
                actions_idx_line = []

                count_lines_gen += 1

            if predicted_action == self.line_separator and count_lines_gen == n_lines_to_gen:
                break
            # all_predictions.append(last_predictions_str)

        if self.args.data_type == "limerick":
            sampled_str = ' | '.join([' '.join(actions_line) for actions_line in actions])
        else:
            sampled_str = ' <eos> '.join([' '.join(actions_line) for actions_line in actions])

        if debug:
            print("[Sample]: sampled_str= ", sampled_str)

        assert len(logprobs) == len(actions_idx), "Length mis-match for logprobs and actions_idx!"
        return {'logprobs': logprobs, 'actions': actions_idx}

    @overrides
    def forward(self,  # type: ignore
                source_tokens: [str, torch.LongTensor],
                target_tokens: [str, torch.LongTensor] = None,
                ending_words: [str, torch.LongTensor] = None,
                batch_idx: int = None,
                decoder_hidden: torch.FloatTensor = None,
                decoder_context: torch.FloatTensor = None,
                ending_words_mask: [str, torch.Tensor] = None,
                hier_mode: bool = False) -> Dict[str, torch.Tensor]:

        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
        # pdb.set_trace()
        # source_mask = utils.get_text_field_mask(source_tokens)

        source_mask = get_text_field_mask(source_tokens)

        embedded_input = self._target_embedder(source_tokens["tokens"])

        targets = target_tokens["tokens"]
        # target_mask = util.get_text_field_mask(target_tokens)
        target_mask = get_text_field_mask(target_tokens)

        batch_size, time_steps, _ = embedded_input.size()

        embedded_ending_words = self._context_embedder(target_tokens["tokens"])

        # pdb.set_trace()
        # Apply dropout to embeddings
        # embedded_input = self._dropout(embedded_input)
        embedded_input = self._lockdropout(x=embedded_input, dropout=self._dropout_ratio)

        if self._sampling_scheme == "first_word":
            if ((not self.training) and np.random.rand() < 0.05):

                i2v = self.vocab

                ending_words_idx = ending_words["tokens"][0].data.cpu().numpy()
                ending_words_str = '|'.join([i2v[int(word)] for word in ending_words_idx])
                print("Ending word sequence : ", ending_words_str)

                start_token_idx = source_tokens["tokens"][0][0]
                self._sample(start_token_idx=start_token_idx, ending_words=ending_words["tokens"][0], debug=True)

        if decoder_hidden is None:
            (decoder_hidden, decoder_context) = self.initialize_hidden(batch_size=batch_size)

        hiddens = []
        contexts = []
        p_gens = []

        for t, emb_t in enumerate(embedded_input.chunk(time_steps, dim=1)):

            decoder_hidden, decoder_context = self._lm_cell(emb_t.squeeze(1), (decoder_hidden, decoder_context))

            hiddens.append(decoder_hidden.unsqueeze(1))
            contexts.append(decoder_context.unsqueeze(1))

        hidden = torch.cat(hiddens, 1)
        # context = torch.cat(contexts, 1)
        output = self._lockdropout(x=hidden, dropout=self._dropout_ratio)

        batch_size = output.size(0)
        seq_len = output.size(1)
        hidden_dim = output.size(2)

        pre_decoded_output = self._intermediate_projection_layer(output.view(batch_size * seq_len, hidden_dim))
        decoded_output = self._output_projection_layer(pre_decoded_output)
        logits = decoded_output.view(batch_size, seq_len, decoded_output.size(1))

        
        class_probabilities = F.softmax(logits, dim=-1)
        _, predicted_classes = torch.max(class_probabilities, dim=-1)

        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": predicted_classes}

        # This code block masks all line endings (ending words)
        if not self.training and hier_mode == True:
            # pdb.set_trace()
            tmp_mask = (1 - ending_words_mask["tokens"])
            target_mask = target_mask.long() * tmp_mask

        loss = self._get_loss_custom(logits, targets, target_mask, training=self.training)

        output_dict["loss"] = loss

        target_mask = get_text_field_mask(target_tokens)
        source_sentence_lengths = get_lengths_from_binary_sequence_mask(mask=source_mask)
        target_sentence_lengths = get_lengths_from_binary_sequence_mask(mask=target_mask)

        output_dict["source_sentence_lengths"] = source_sentence_lengths
        output_dict["target_sentence_lengths"] = target_sentence_lengths

        # if self.training:
        decoder_hidden = []
        decoder_context = []

        for idx, length in enumerate(source_sentence_lengths):
            assert source_sentence_lengths[idx] == target_sentence_lengths[idx], "Mis-match!"
            decoder_hidden.append(hiddens[length - 1][idx].squeeze(0))
            decoder_context.append(contexts[length - 1][idx].squeeze(0))

        output_dict["decoder_hidden"] = decoder_hidden
        output_dict["decoder_context"] = decoder_context

        return output_dict

    def get_context(self, batch_size, ending_words):
        assert False, "No longer used - marked for removal"
        # if self._encoder_type is not None:
        #     ending_words_embedded = self._context_embedder(ending_words)
        #     encoder_hidden, encoder_context = self.initialize_hidden(batch_size=batch_size,
        #                                                              hidden_dim=self._encoder_hidden_size, n_dim=3)
        #     _, (embedded_ending_words, _) = self._encoder(ending_words_embedded, (encoder_hidden, encoder_context))
        #     embedded_ending_words = embedded_ending_words.squeeze(0)
        # elif self._context_embedder is not None:
        #     embedded_ending_words = torch.sum(self._context_embedder(ending_words), dim=1)
        # else:
        #     embedded_ending_words = torch.sum(self._target_embedder(ending_words), dim=1)

        # return embedded_ending_words

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        if not self.training:
            metrics.update(self._sonnet_eval.get_metric(reset))

        return metrics

    @staticmethod
    def _get_loss_custom(logits: torch.LongTensor,
                         targets: torch.LongTensor,
                         target_mask: torch.LongTensor,
                         training: bool = True) -> torch.LongTensor:
        """
        As opposed to get_loss, logits and targets are of same size
        """
        relevant_targets = targets.contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask.contiguous()  # (batch_size, num_decoding_steps)
        # loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

        if training:
            loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        else:
            loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask,
                                                           average=None)

        return loss

    
