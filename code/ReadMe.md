

#### Pretraining Discriminator's Encoder

```
python main.py  --solver_type Endings --model_name rhymgan_ae1 --use_alignment false  --model_type encdec --data_type ae  | tee logs/rhymgan_ae1_train.log
```

Set 'use_cuda' as false to run without gpu

#### LM training


```
python main.py --solver_type Main --mode train_lm --model_name rhymgan_lm1  --load_gutenberg true  --emsize 100  --data_type sonnet_endings --vanilla_lm_path tmp/rhymgan_lm1/ | tee logs/rhymgan_lm1_train.log
```

```
python main.py --solver_type Main --mode train_lm --model_name rhymgan_limerick_lm1  --load_gutenberg true  --emsize 100  --data_type limerick --vanilla_lm_path tmp/rhymgan_limerick_lm1/ > logs/rhymgan_limerick_lm1_train.log
```

#### RHYMEGAN

###### RhymeGAN training
```
#---- Sonnet
python main.py --solver_type Main --model_name rhymgan1 --vanilla_lm_path tmp/rhymgan_lm1/ --load_vanilla_lm true --g2p_model_name rhymgan_ae1 --trainable_g2p true --reinforce_weight=0.1 --use_all_sonnet_data true --num_samples_at_epoch_end 150 --epochs 80  > logs/rhymgan1_train.log
```

```
#---- Sonnet2
python main.py --solver_type Main --model_name rhymgan2 --vanilla_lm_path tmp/rhymgan_lm1/ --load_vanilla_lm true --g2p_model_name rhymgan_ae1 --trainable_g2p true --reinforce_weight=0.1 --use_all_sonnet_data true --num_samples_at_epoch_end 150 --epochs 80  > logs/rhymgan2_train.log
```



```
#---- Limerick
python main.py --solver_type Main --model_name rhymgan_limerick1  --data_type limerick --vanilla_lm_path tmp/rhymgan_limerick_lm1/ --load_vanilla_lm true --g2p_model_name rhymgan_ae1 --trainable_g2p true --reinforce_weight=0.1 --use_all_sonnet_data true --num_samples_at_epoch_end 150 --epochs 70  > logs/rhymgan_limerick1_train.log

```

###### Eval

TODO - Code Documentation in Progress
```
#---- Sonnet
EPOCH_TO_TEST=69
#eval
MODEL_NAME=rhymgan1
python main.py --solver_type Main --model_name $MODEL_NAME --vanilla_lm_path tmp/rhymgan_lm1/ --load_vanilla_lm true --g2p_model_name rhymgan_ae1 --trainable_g2p true --reinforce_weight=0.1 --use_all_sonnet_data true --num_samples_at_epoch_end 100 --epochs 80 --mode eval --epoch_to_test $EPOCH_TO_TEST  | tee  logs/"$MODELNAME"_eval.log
#eval with lower temperature
CUDA_VISIBLE_DEVICES=1 python main.py --solver_type Main --model_name $MODEL_NAME --vanilla_lm_path tmp/rhymgan_lm1/ --load_vanilla_lm true --g2p_model_name rhymgan_ae1 --trainable_g2p true  --use_all_sonnet_data true --mode eval --epoch_to_test $EPOCH_TO_TEST --temperature 0.7 | tee logs/"$MODEL_NAME"_temp7_eval.log
```