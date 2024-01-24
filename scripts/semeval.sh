kshot=1-5

CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs=40  --num_workers=8 \
    --model_name_or_path roberta-large \
    --config roberta-large\
    --accumulate_grad_batches 1 \
    --batch_size 8 \
    --data_dir dataset/semeval/k-shot/${kshot} \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --use_template_words 0 \
    --init_type_words 0 \
    --init_answer_words 1 \
    --contrastive_ratio 2.0 \
    --contrastive_beta 0.1 \
    --use_contrastive \
    --pipeline_init \
    --output_dir output/semeval/k-shot/$kshot \
    --MVRE \
    --multi_viewer \
    --multi_viewer_num 3