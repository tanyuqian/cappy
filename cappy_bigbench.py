from functools import partial
import os
import json
import random
import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rouge_score.rouge_scorer import RougeScorer
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
from redco import JsonlDataset, Deployer, Trainer


def get_cappy_dataset(bigbench_gens,
                      subset_name,
                      src_key,
                      tgt_key,
                      refs_key,
                      label_key,
                      train_size):
    flag_set, tgt_set = set(), set()
    train_examples = []
    for example in bigbench_gens[f'{subset_name}_train']:
        for ref in example[refs_key]:
            train_examples.append(
                {src_key: example[src_key], tgt_key: ref, label_key: 1.})
            flag_set.add(f'{example[src_key]}|||{ref}')
            tgt_set.add(ref)

    neg_examples = []
    for example in train_examples:
        src, tgt = example[src_key], random.choice(sorted(list(tgt_set)))
        if f'{src}|||{tgt}' not in flag_set:
            neg_examples.append({src_key: src, tgt_key: tgt, label_key: 0.})
            flag_set.add(f'{src}|||{tgt}')
    train_examples.extend(neg_examples)

    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    for example in bigbench_gens[f'{subset_name}_train']:
        src, refs = example[src_key], example[refs_key]
        for decoding in example['flan_samples']:
            for sample in example['flan_samples'][decoding]:
                tgt = sample[tgt_key]
                if f'{src}|||{tgt}' not in flag_set:
                    flag_set.add(f'{src}|||{tgt}')
                else:
                    continue

                rouge_l = rouge_scorer.score_multi(
                    targets=refs, prediction=tgt)['rougeL'].fmeasure
                train_examples.append(
                    {src_key: src, tgt_key: tgt, label_key: rouge_l})

    random.shuffle(train_examples)
    train_examples = \
        (train_examples * (train_size // len(train_examples) + 1))[:train_size]

    validation_examples = []
    for example_idx, example in enumerate(
            bigbench_gens[f'{subset_name}_validation']):
        for decoding in example['flan_samples']:
            for sample in example['flan_samples'][decoding]:
                validation_examples.append({
                    'example_idx': example_idx,
                    src_key: example[src_key],
                    tgt_key: sample[tgt_key]
                })

    return {'train': train_examples, 'validation': validation_examples}


def collate_fn(examples, src_key, tgt_key, label_key, tokenizer, max_length):
    batch = tokenizer(
        [(example[src_key], example[tgt_key]) for example in examples],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    if label_key in examples[0]:
        batch['labels'] = np.array([example[label_key] for example in examples])

    return batch


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop('labels')
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training).logits

    return jnp.mean(jnp.square(logits[..., 0] - labels))


def pred_fn(pred_rng, params, batch, model):
    if 'labels' in batch:
        batch.pop('labels')
    return model(**batch, params=params, train=False).logits[..., 0]


def eval_rouge(examples, preds, refs_key):
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)

    scores = []
    for example, hypo in zip(examples, preds):
        scores.append(rouge_scorer.score_multi(
            targets=example[refs_key], prediction=hypo)['rougeL'].fmeasure)

    return np.mean(scores)


def main(bigbench_gen_dir='bigbench_flan_gens',
         bigbench_gen_model='flan-t5-xxl',
         bigbench_subset_name='auto_categorization',
         src_key='instruction',
         tgt_key='response',
         refs_key='references',
         label_key='label',
         model_name_or_path='btan2/cappy-large',
         n_model_shards=1,
         max_length=512,
         train_size=102400,
         per_device_batch_size=8,
         eval_per_device_batch_size=32,
         accumulate_grad_batches=16,
         learning_rate=2e-5,
         warmup_rate=0.1,
         weight_decay=0.,
         results_dir='bigbench_cappy_results',
         seed=11111):
    result_filename = \
        f'{results_dir}/{bigbench_gen_model}/{bigbench_subset_name}.json'
    if os.path.exists(result_filename):
        print(f'Result already exists in {result_filename}. Skipped.')
        return
    else:
        os.makedirs(f'{results_dir}/{bigbench_gen_model}', exist_ok=True)
        open(result_filename, 'w').write('running...')

    deployer = Deployer(n_model_shards=n_model_shards, jax_seed=seed)
    random.seed(seed)

    bigbench_gens = JsonlDataset(
        data_dir=f'{bigbench_gen_dir}/{bigbench_gen_model}')

    with jax.default_device(jax.devices('cpu')[0]):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = FlaxAutoModelForSequenceClassification.from_pretrained(
            model_name_or_path)
        model.params = model.to_fp32(model.params)

    lr_schedule_fn = deployer.get_lr_schedule_fn(
        train_size=train_size,
        per_device_batch_size=per_device_batch_size,
        n_epochs=1,
        learning_rate=learning_rate,
        schedule_type='linear',
        warmup_rate=warmup_rate)
    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=lr_schedule_fn, weight_decay=weight_decay),
        every_k_schedule=accumulate_grad_batches)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            src_key=src_key,
            tgt_key=tgt_key,
            label_key=label_key,
            tokenizer=tokenizer,
            max_length=max_length),
        apply_fn=model,
        loss_fn=loss_fn,
        params=model.params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        accumulate_grad_batches=accumulate_grad_batches,
        params_sharding_rules=deployer.get_sharding_rules(params=model.params))

    predictor = trainer.get_default_predictor(
        pred_fn=partial(pred_fn, model=model))

    cappy_dataset = get_cappy_dataset(
        bigbench_gens=bigbench_gens,
        subset_name=bigbench_subset_name,
        src_key=src_key,
        tgt_key=tgt_key,
        refs_key=refs_key,
        label_key=label_key,
        train_size=train_size)

    trainer.train(
        examples=cappy_dataset['train'],
        per_device_batch_size=per_device_batch_size)

    cappy_scores = predictor.predict(
        examples=cappy_dataset['validation'],
        per_device_batch_size=eval_per_device_batch_size,
        params=trainer.params,
        params_meshed=(n_model_shards > 1))

    best_scores = \
        [float('-inf') for _ in range(len(cappy_dataset['validation']))]
    preds = [None for _ in range(len(cappy_dataset['validation']))]
    for example, cappy_score in zip(
            cappy_dataset['validation'], cappy_scores):
        example_idx = example['example_idx']
        if cappy_score > best_scores[example_idx]:
            best_scores[example_idx] = cappy_score
            preds[example_idx] = example[tgt_key]

    result = eval_rouge(
        examples=bigbench_gens[f'{bigbench_subset_name}_validation'],
        preds=preds,
        refs_key=refs_key)
    json.dump({'rougeL': result}, open(result_filename, 'w'))

    print(f'Cappy + {bigbench_gen_model} on {bigbench_subset_name}: {result}')


if __name__ == '__main__':
    fire.Fire(main)