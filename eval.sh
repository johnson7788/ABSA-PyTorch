! cd ABSA-PyTorch/ && python train.py --model_name lcf_bert --dataset cosmetics --max_seq_len 200 --valset_ratio 0.1 --pretrained_bert_name bert-base-chinese --pretrained_bert_cache_dir model_cache --do_eval --eval_model_path /content/ABSA-PyTorch/absa_bert/lcf_bert_cosmetics_step770_val_acc0.8325
! cd ABSA-PyTorch/ && python train.py --model_name aen_bert --dataset cosmetics --max_seq_len 183 --valset_ratio 0.1 --pretrained_bert_name bert-base-chinese --pretrained_bert_cache_dir model_cache --do_eval --eval_model_path /content/ABSA-PyTorch/absa_bert/aen_bert_cosmetics_val_acc0.7857
! cd ABSA-PyTorch/ && python train.py --model_name bert_spc --dataset cosmetics --max_seq_len 183 --valset_ratio 0.1 --pretrained_bert_name bert-base-chinese --pretrained_bert_cache_dir model_cache --do_eval --eval_model_path /content/ABSA-PyTorch/absa_bert/bert_spc_cosmetics_val_acc0.8369

! cd ABSA-PyTorch/ && python train.py --model_name bert_spc --dataset cosmetics --max_seq_len 65 --valset_ratio 0.1 --pretrained_bert_name bert-base-chinese --pretrained_bert_cache_dir model_cache --do_eval --eval_model_path /content/ABSA-PyTorch/absa_bert/bert_spc_cosmetics_step1600_val_acc0.8424_truncate

