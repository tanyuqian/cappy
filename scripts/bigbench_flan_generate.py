import os
from functools import partial
import fire
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM
from redco import JsonlDataset, Deployer, Predictor


def collate_fn(examples, tokenizer, max_src_len, src_key):
    return tokenizer(
        [example[src_key] for example in examples],
        max_length=max_src_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')


def log_likelihood_collate_fn(examples,
                              tokenizer,
                              decoder_start_token_id,
                              max_src_len,
                              max_tgt_len,
                              src_key,
                              tgt_key):
    model_inputs = tokenizer(
        [example[src_key] for example in examples],
        max_length=max_src_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    decoder_inputs = tokenizer(
        [example[tgt_key] for example in examples],
        max_length=max_tgt_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    model_inputs['labels'] = np.copy(decoder_inputs['input_ids'])
    decoder_inputs['input_ids'][:, 1:] = decoder_inputs['input_ids'][:, :-1]
    decoder_inputs['input_ids'][:, 0] = decoder_start_token_id

    for key in decoder_inputs:
        model_inputs[f'decoder_{key}'] = np.array(decoder_inputs[key])

    return model_inputs


def pred_fn(pred_rng, params, batch, model, gen_kwargs):
    return model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        prng_key=pred_rng,
        **gen_kwargs).sequences


def log_likelihood_pred_fn(pred_rng, params, batch, model):
    labels, label_weights = batch.pop('labels'), batch['decoder_attention_mask']
    logits = model(**batch, params=params, train=False)[0]

    log_likelihoods = -optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)
    ll_sum = jnp.sum(log_likelihoods * label_weights, axis=-1)
    ll_mean = ll_sum / jnp.sum(label_weights, axis=-1)

    return {'log_likelihood_sum': ll_sum, 'log_likelihood_mean': ll_mean}


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def main(data_dir='bigbench_data',
         src_key='instruction',
         tgt_key='response',
         model_name_or_path='google/flan-t5-small',
         n_model_shards=1,
         n_sample_gens=4,
         per_device_batch_size=8,
         max_src_len=512,
         max_tgt_len=64,
         output_dir='bigbench_flan_gens',
         jax_seed=42):
    deployer = Deployer(n_model_shards=n_model_shards, jax_seed=jax_seed)

    with jax.default_device(jax.devices('cpu')[0]):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        model.params = model.to_fp16(model.params)
        params_sharding_rules = deployer.get_sharding_rules(model.params)

    gen_length_kwargs = {'min_length': 3, 'max_length': max_tgt_len}
    gen_kwargs = {
        'beam4': {'num_beams': 4, **gen_length_kwargs},
        'sampling': {'do_sample': True, **gen_length_kwargs},
        'temp0.9': {'do_sample': True, 'temperature': 0.9, **gen_length_kwargs},
        'topk40': {'do_sample': True, 'top_k': 40, **gen_length_kwargs},
        'topp0.95': {'do_sample': True, 'top_p': 0.95, **gen_length_kwargs},
    }

    predictors = {
        gen_key: Predictor(
            deployer=deployer,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                max_src_len=max_src_len,
                src_key=src_key),
            pred_fn=partial(
                pred_fn, model=model, gen_kwargs=gen_kwargs[gen_key]),
            output_fn=partial(output_fn, tokenizer=tokenizer),
            params_sharding_rules=params_sharding_rules
        ) for gen_key in gen_kwargs.keys()
    }

    log_likelihood_predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            log_likelihood_collate_fn,
            tokenizer=tokenizer,
            decoder_start_token_id=model.config.decoder_start_token_id,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            src_key=src_key,
            tgt_key=tgt_key),
        pred_fn=partial(log_likelihood_pred_fn, model=model),
        params_sharding_rules=params_sharding_rules)

    dataset = JsonlDataset(data_dir=data_dir)
    for subset_name in json.load(open(f'{data_dir}/subset_names.json')):
        for split in ['train', 'validation']:
            examples = dataset[f'{subset_name}_{split}']
            for example in examples:
                example['flan_samples'] = \
                    {gen_key: [] for gen_key in gen_kwargs}

            for gen_key in gen_kwargs:
                for sample_idx in range(
                        n_sample_gens if gen_key != 'beam4' else 1):
                    desc = f'{subset_name}_{split}_{gen_key}_{sample_idx}'
                    preds = predictors[gen_key].predict(
                        examples=examples,
                        params=model.params,
                        per_device_batch_size=per_device_batch_size,
                        desc=desc)

                    if split == 'validation':
                        ll_examples = [
                            {src_key: example[src_key], tgt_key: pred}
                            for example, pred in zip(examples, preds)
                        ]
                        log_likelihoods = log_likelihood_predictor.predict(
                            examples=ll_examples,
                            params=model.params,
                            per_device_batch_size=per_device_batch_size,
                            desc=f'{desc}_log_likelihood')
                        log_likelihoods = jax.tree_util.tree_map(
                            lambda x: x.item(), log_likelihoods)
                    else:
                        log_likelihoods = [{} for _ in range(len(examples))]

                    for example, pred, log_likelihood in zip(
                            examples, preds, log_likelihoods):
                        example['flan_samples'][gen_key].append(
                            {tgt_key: pred, **log_likelihood})

            file_dir = f'{output_dir}/' + model_name_or_path.split('/')[-1]
            filename = f'{subset_name}_{split}.jsonl'
            os.makedirs(file_dir, exist_ok=True)
            with open(f'{file_dir}/{filename}', 'w') as output_file:
                for example in examples:
                    output_file.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    fire.Fire(main)
