from functools import partial
import json
import fire
import jax
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
import datasets
from redco import Deployer, Predictor


def collate_fn(examples, sent0_key, sent1_key, tokenizer, max_length):
    return tokenizer(
        [(example[sent0_key], example[sent1_key]) for example in examples],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np')


def pred_fn(pred_rng, batch, params, model):
    return model(**batch, params=params, train=False).logits[..., 0]


def eval_metric_fn(examples, preds):
    task_preds, task_labels = {}, {}

    for example, pred in zip(examples, preds):
        example_key = '|||'.join([
            example['dataset_name'],
            example['template_name'],
            str(example['example_idx'])])

        if task_preds.get(example_key, (None, float('-inf')))[1] < pred:
            task_preds[example_key] = (example['response'], pred)
        if example['label'] == 1:
            task_labels[example_key] = example['response']

    n_correct, n_total = {}, {}
    for example_key in task_labels.keys():
        dataset_name = example_key.split('|||')[0]

        n_correct[dataset_name] = n_correct.get(dataset_name, 0) + int(
            task_preds[example_key][0] == task_labels[example_key])

        n_total[dataset_name] = n_total.get(dataset_name, 0) + 1

    return {
        dataset_name: n_correct[dataset_name] / n_total[dataset_name]
        for dataset_name in n_total.keys()
    }


def main(data_file='./promptsource_test.jsonl',
         sent0_key='instruction',
         sent1_key='response',
         model_name_or_path='btan2/cappy-large',
         n_model_shards=1,
         per_device_batch_size=128,
         max_length=512,
         jax_seed=42):
    deployer = Deployer(jax_seed=jax_seed, n_model_shards=n_model_shards)

    with jax.default_device(jax.devices('cpu')[0]):
        test_examples = list(datasets.load_dataset(
            'json', data_files={'test': data_file}, split='test'))
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = FlaxAutoModelForSequenceClassification.from_pretrained(
            model_name_or_path)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            sent0_key=sent0_key,
            sent1_key=sent1_key,
            tokenizer=tokenizer,
            max_length=max_length),
        pred_fn=partial(pred_fn, model=model),
        params_sharding_rules=deployer.get_sharding_rules(params=model.params))

    preds = predictor.predict(
        examples=test_examples,
        params=model.params,
        per_device_batch_size=per_device_batch_size)

    results = eval_metric_fn(examples=test_examples, preds=preds)
    results['Average'] = sum(results.values()) / len(results.values())

    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    fire.Fire(main)
