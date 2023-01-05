import torch
import argparse
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import os
import numpy as np

from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, load_metric, Dataset, DatasetDict, Features, Sequence, Value

import dataloader as DataProcess
import model as Model

import utils.metric as Metric

import retrieval as Retrieval
from utils.seed_setting import seed_setting

def main(config):
    seed_setting(config.train.seed)
    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    device = torch.device('cuda')

    print('='*50,f'현재 적용되고 있는 전처리 클래스는 {config.data.preprocess}입니다.', '='*50, sep='\n\n')
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    prepare_features = getattr(DataProcess, config.data.preprocess)(tokenizer, config.train.max_length, config.train.max_answer_length, config.train.stride)
    test_data = load_from_disk(config.data.test_path)

    retrieval = getattr(Retrieval, config.retrieval.retrieval_class)(
            tokenizer = tokenizer,
            data_path=config.retrieval.retrieval_path,
            context_path = config.retrieval.retrieval_data,
            is_faiss = config.retrieval.is_faiss
            )
    test_wiki_data = retrieval.retrieve(query_or_dataset=test_data, topk = config.retrieval.topk)['validation']

    test_dataset = test_wiki_data.map(
            prepare_features.test,
            batched=True,
            num_proc=1,
            remove_columns=test_wiki_data.column_names,
            load_from_cache_file=True,
            )

    metric = getattr(Metric, config.model.metric_class)(
                metric = load_metric('squad'),
                dataset = test_dataset,
                raw_data = test_wiki_data,
                n_best_size = config.train.n_best_size,
                max_answer_length = config.train.max_answer_length,
                save_dir = config.save_dir,
                mode = 'test',
                tokenizer = tokenizer
            )

    test_dataset = test_dataset.remove_columns(["example_id", "offset_mapping"])
    test_dataset.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, collate_fn=data_collator, pin_memory=True, shuffle=False)
  
    print('='*50,f'현재 적용되고 있는 모델 클래스는 {config.model.model_class}입니다.', '='*50, sep='\n\n')
    model = getattr(Model, config.model.model_class)(
        model_name = config.model.model_name,
        num_labels=2,
        dropout_rate = config.train.dropout_rate,
        ).to(device)

    best_model = [model for model in os.listdir(f'./save/{config.save_dir}') if 'nbest' not in model and 'best' in model][0]
    checkpoint = torch.load(f'./save/{config.save_dir}/{best_model}')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    len_val_dataset = test_dataloader.dataset.num_rows
    with torch.no_grad():
        all_preds = []
        for test_batch in tqdm(test_dataloader):
            inputs = {
                'input_ids': test_batch['input_ids'].to(device),
                'attention_mask': test_batch['attention_mask'].to(device),
            }

            pred_ids = model.model.generate(
                    input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    max_length = config.train.max_answer_length,
                    do_sample=True,
                    top_p=0.92, 
                    top_k=30,
                    temperature=0.9,
            )
            pred_ids = pred_ids.cpu().numpy()
            for pred_id in pred_ids:
                pred_decoded = tokenizer.decode(pred_id)
                all_preds.append(pred_decoded)

    metrics = metric.gen_compute_EM_f1(all_preds, None)

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    args, _ = parser.parse_known_args()
    ## ex) python3 train.py --config baseline
    
    config = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')

    main(config)