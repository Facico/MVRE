kshot=1-1

CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs=20  --num_workers=8 \
    --model_name_or_path roberta-large \
    --config roberta-large\
    --data_type tacrev \
    --accumulate_grad_batches 1 \
    --batch_size 8 \
    --dev_batch_size 8 \
    --data_dir dataset/tacrev/k-shot/$kshot \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 512 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --use_template_words 0 \
    --init_type_words 0 \
    --init_answer_words 1 \
    --output_dir output/tacrev/k-shot/$kshot \
    --contrastive_ratio 2.0 \
    --contrastive_beta 0.1 \
    --pipeline_init \
    --use_contrastive \
    --MVRE \
    --multi_viewer \
    --multi_viewer_num 3