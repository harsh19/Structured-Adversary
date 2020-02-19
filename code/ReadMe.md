

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

TODO - Code Documentation in Progress
