from constants import *
import torch
import random
import numpy as np
import argparse
def str2bool(val):
    if val.lower() in ['1','true','t']:
        return True
    if val.lower() in ['0','false','f']:
        return False
    print("val = ", val)
    return 0/0

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="train",help='train/train_lm')
parser.add_argument('--use_cuda', type=str2bool, default='true')
parser.add_argument('--model_name', type=str, default="default",help='')
parser.add_argument('--use_alignment', type=str2bool, default="false",help='')
parser.add_argument('--model_type', type=str, default="encdec",help='')
parser.add_argument('--data_type', type=str, default="sonnet_endings",help='limerick or sonnet_endings')
parser.add_argument('--use_all_sonnet_data', type=str2bool, default="false",help='')
parser.add_argument('--debug', type=str2bool, default="false",help='')
parser.add_argument('--epochs', type=int, default=41,help='')
parser.add_argument('--batch_size', type=int, default=64,help='')
parser.add_argument('--gutenberg_emb_path', default="data/pretrained_embeddings/gutenberg_word2vec.txt",help='')
parser.add_argument('--sonnet_vocab_path', default="data/splits/sonnet/vocabulary_v2/tokens.txt",help='')
parser.add_argument('--limerick_vocab_path', default="data/vocab/limerick_vocabulary_0_threshold_2_tokens.txt",help='')
parser.add_argument('--cmu_data_path', default="data/resources/cmudict-0.7b.txt",help='')
parser.add_argument('--sonnet_data_path', default="data/splits/sonnet/sonnet_",help='')
parser.add_argument('--all_line_endings_sonnet', default="data/splits/sonnet/all_line_endings_sonnet_quatrains_valid.json",help='')
parser.add_argument('--all_line_endings_sonnet_test', default="data/splits/sonnet/all_line_endings_sonnet_quatrains_test.json",help='')
parser.add_argument('--all_line_endings_limerick', default="data/splits/limerick_only_subset/all_line_endings_limerick_val.json",help='')
parser.add_argument('--all_line_endings_limerick_test', default="data/splits/limerick_only_subset/all_line_endings_limerick_test.json",help='')
parser.add_argument('--limerick_data_path', default="data/splits/limerick_only_subset/",help='')
parser.add_argument('--tie_emb', type=str2bool, default="false",help='')
parser.add_argument('--use_eow_in_enc', type=str2bool, default="false",help='MAKE SURE this is consistent with option used in g2p training. this bascally attaches a end marker at of seq of charatcters in a word. This should impoact discriminators ugin charatccer level encoders')


parser.add_argument('--pretraining_data_type', default='cmu_dict_words', help='cmu_dict_words OR sonnet OR limerick')
parser.add_argument('--g2p_model_name', default=None,help='')
parser.add_argument('--use_reinforce_loss', default='true', type=str2bool, help='')
parser.add_argument('--trainable_g2p', default='true', type=str2bool, help='')
parser.add_argument('--reinforce_weight', default=1.0, type=float, help='')
parser.add_argument('--load_gutenberg', default='false', type=str2bool, help='')
parser.add_argument('--load_gutenberg_path', default="data/pretrained_embeddings/gutenberg_word2vec.txt", help='')
parser.add_argument('--freeze_emb', default='false', type=str2bool, help='')
parser.add_argument('--emsize', default=128, type=int, help='')
parser.add_argument('--H', default=128, type=int, help='')
parser.add_argument('--pretrain_lm', default='false', type=str2bool, help='')
parser.add_argument('--train_lm_supervised', default='false', type=str2bool, help='train lm using superivsed objective also')
parser.add_argument('--add_entropy_regularizer',  default='false', type=str2bool, help='')  #TODO - not yet implemented
parser.add_argument('--use_score_as_reward', default='false', type=str2bool, help='')
parser.add_argument('--solver_type', default='Endings', type=str, help='Main, Endings')
parser.add_argument('--seed', default=123, type=int, help='')
parser.add_argument('--save_vanilla_lm', default='true', type=str2bool, help='Whether to save vanilla LM')
parser.add_argument('--load_vanilla_lm', default='false', type=str2bool, help='Whether to load pre-trained vanilla LM')
parser.add_argument('--vanilla_lm_path', default='tmp/tmp_best_vanilla_lm', help='Path to folder which has the stored vanilla lm')

#analysis mode params only
parser.add_argument('--epoch_to_test', default='40', help='')
parser.add_argument('--tmp_dir', default='tmp/', help='')
parser.add_argument('--dump_matrices', default='true', type=str2bool, help='dump_matrices')
parser.add_argument('--learn_g2p_encoder_from_scratch', default='false',type=str2bool,help='') ## when true, does NOT load g2pmodel from g2p_model_name, though still loads indexers from g2p_model_name
parser.add_argument('--temperature', default=1.0, type=float, help='currently being used only in eval mode while generation') ##
parser.add_argument('--disc_type', default=DISC_TYPE_MATRIX, type=str, help=''+DISC_TYPE_MATRIX+' OR '+DISC_TYPE_NON_STRUCTURED) ##
parser.add_argument('--num_samples_at_epoch_end', default=40, type=int, help='') ##

args = parser.parse_args()
print(" ======== args ====== ")
for arg in vars(args):
    print( arg, ":", getattr(args,arg) )
print("============== \n ")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
use_cuda = cuda = args.use_cuda
if args.load_gutenberg:
    assert args.emsize==100, "Gutenberg embeddings are of size 100 - use emsize=100"
print()


from solvers_merged import *
from solvers_pretrain_disc_encoder import *
# from utils import Indexer

assert args.disc_type in [DISC_TYPE_MATRIX, DISC_TYPE_NON_STRUCTURED]

################  Data

vocab = None
if args.data_type == 'limerick':
    limerick_vocab = load_sonnet_vocab(args.limerick_vocab_path)
    print(len(limerick_vocab), limerick_vocab[0:5])
    vocab = limerick_vocab
else:
    print("[INFO] Loading Sonnet vocab ... ")
    sonnet_vocab = load_sonnet_vocab(args.sonnet_vocab_path)
    print("len(sonnet_vocab), sonnet_vocab[0:5] = ", len(sonnet_vocab), sonnet_vocab[0:5])
    vocab = sonnet_vocab
    print()


print("[INFO] Loading CMU dictionary...")
cmu_data = loadCMUDict(args.cmu_data_path)
print("cmu dictionary: list(cmu_data.items())[0:10] = ", list(cmu_data.items())[0:10])
print()


################  Model dumps save dir
model_dir = 'tmp/tmp_'+args.model_name+'/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
args.model_dir = model_dir

################  Solver choice
if args.solver_type == "Main":   # endings model training, LM training

    solver = MainSolver(typ=args.data_type, cmu_dict=cmu_data, args=args, mode=args.mode)

    if args.mode == "train_lm":

        if not os.path.exists(args.vanilla_lm_path):
            os.makedirs(args.vanilla_lm_path)

        solver.train_lm(epochs=30, debug=False, args=args)
        # solver.train_lm(epochs=1, debug=True, args=args)

    else:

        #### Loading pretrained LM
        if args.load_vanilla_lm:
            lm_model_state_dict = torch.load(os.path.join(args.vanilla_lm_path, 'model_best'))
            solver.lm_model.load_state_dict(lm_model_state_dict)
            print("LM model initialized with pre-trained model!")

        ################ Solver sanity check and Save solver indexers, etc
        # - only if train mode (to not overwrite in analysis/eval mode)
        solver.get_stats_ending_words(split='train', batch_size=32)
        print(solver.g_indexer.w_cnt)

        if args.mode=="train":
            solver.save(model_dir + 'solver_')
        else:
            solver.load(model_dir + 'solver_')

        # Some sanity checks
        # list(solver.g_indexer.w2idx.items())[0:5]
        x_sample,_,x_start = solver.get_batch(i=0, batch_size=3, split='train') #, typ=args.data_type)
        print("x_sample = ", x_sample)


        ################ Pretraining LM Model
        if args.pretrain_lm:
            args.use_reinforce_loss = False
            model_dir = args.model_dir
            model_name = args.model_name
            args.model_dir = 'tmp/tmp_pretrain_'+args.model_name+'/'
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
            args.model_name = 'pretrain_'+args.model_name
            solver.train(args.epochs, debug=args.debug)
            state_dict_best = torch.load(args.model_dir+'model_best')
            solver.model.load_state_dict(state_dict_best)
            #reset
            args.use_reinforce_loss = True
            args.model_dir = model_dir
            args.model_name = model_name


        ################ mode=train:Training Full Model
        print(args.mode)
        if args.mode=="train":
            print("-x"*55)
            solver.train(args.epochs, debug=args.debug, train_lm_supervised=args.train_lm_supervised)
        elif args.mode=="eval": # || mode=eval  eval_type=rhyming
            # load the model states
            # solver.load_models(args.model_name, args.model_state)
            args.model_dir = args.tmp_dir+'tmp_'+args.model_name+'/'
            epoch=args.epoch_to_test #'40' #'best' #'40'
            solver.load_models(model_dir=args.model_dir, model_epoch=epoch, load_lm=False)
            #run the analysis
            solver.analysis(epoch=epoch, args=args)
        elif args.mode == "lm_eval":
            # load the model states
            # solver.load_models(args.model_name, args.model_state)
            print("lm_eval mode ON!!!!!")
            args.model_dir = args.tmp_dir + 'tmp_' + args.model_name + '/'
            epoch = args.epoch_to_test  # '40' #'best' #'40'
            solver.load_models(model_dir=args.model_dir, model_epoch=epoch, load_lm=False)
            # run the analysis
            solver.lm_analysis(epoch=epoch, args=args)


elif args.solver_type == "Endings":   #pretraining

    assert args.data_type in ["ae","g2p", "sonnet_endings", "g2plast"]
    if args.tie_emb:
        assert args.data_type=="ae"
    use_cuda = cuda = args.use_cuda
            
    ################ 
    model_dir = 'tmp/tmp_'+args.model_name+'/'
    if not os.path.exists(model_dir):
        print("Creating ", model_dir, " ... ")
        os.makedirs(model_dir)
        print()

    print("[INFO] Creating solver ... ")
    solver = EndingsSolver(typ=args.data_type, cmu_dict=cmu_data, args=args)
    print()

    print("[INFO] Getting splits ... ")
    # solver.get_splits()
    solver.get_splits(data_type=args.pretraining_data_type) # 'sonnet', 'cmu_dict_words'
    print()

    print("Indexing data ... ")
    solver.index()
    print("[INFO] solver.g_indexer.w_cnt = ", solver.g_indexer.w_cnt)
    print()

    if args.mode=="train":
        print("[INFO] Saving indexer at " , model_dir + 'solver_', " ... ")
        solver.save(model_dir + 'solver_')
        print()

    print("[INFO] Some data samples ... ")
    x_sample,y_sample,y_start = solver.get_batch(i=0, batch_size=3, split='train') #, typ=args.data_type)
    print("x_sample = ", x_sample)
    print("y_sample = ", y_sample)
    print("x_sample[0]: idx_to_2: = ", solver.g_indexer.idx_to_2(x_sample[0]))

    print("[INFO] Create model ... ")
    solver.init_model(y_start=y_start)
    print()

    if args.mode=="train":
        print("[INFO] Beginning Training ... ")
        solver.train(args.epochs, debug=args.debug, use_alignment=args.use_alignment, model_dir=model_dir)

    elif args.mode=="eval":
        if args.data_type=="ae" or args.data_type=="g2plast" or args.data_type=="g2p" or args.data_type=="aelast":
            solver.analyze(vocab=vocab)
        else:
            assert False

else:

    assert False


