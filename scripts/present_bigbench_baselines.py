import json
import fire
import tqdm
import numpy as np
from rouge_score import rouge_scorer
from pandas import DataFrame

FLAN_SIZES = ['small', 'base', 'large', 'xl', 'xxl']
DECODINGS = ['beam4', 'sampling', 'temp0.9', 'topk40', 'topp0.95']


def eval_rouge(examples, preds, refs_key):
    rouger = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = []
    for example, hypo in zip(examples, preds):
        scores.append(rouger.score_multi(
            targets=example[refs_key], prediction=hypo)['rougeL'].fmeasure)

    return np.mean(scores)


def main(bigbench_data_dir='bigbench_data',
         bigbench_gen_dir='bigbench_flan_gens',
         tgt_key='response',
         refs_key='references'):
    results = {}
    for flan_size in FLAN_SIZES:
        model_name = f'flan-t5-{flan_size}'
        model_results = {decoding: [] for decoding in DECODINGS}
        model_results['self_score_sum'] = []
        model_results['self_score_mean'] = []

        for subset_name in tqdm.tqdm(json.load(
                open(f'{bigbench_data_dir}/subset_names.json')
        ), desc=f'Evaluating {model_name}'):
            validation_filename = (f'{bigbench_gen_dir}/{model_name}/'
                                   f'{subset_name}_validation.jsonl')
            examples = [json.loads(line) for line in open(validation_filename)]

            for decoding in DECODINGS:
                preds = [
                    example['flan_samples'][decoding][0][tgt_key]
                    for example in examples
                ]
                model_results[decoding].append(eval_rouge(
                    examples=examples, preds=preds, refs_key=refs_key))

            for aggr in ['mean', 'sum']:
                preds = []
                for example in examples:
                    best_score, pred = float('-inf'), None
                    for decoding in DECODINGS:
                        for flan_sample in example['flan_samples'][decoding]:
                            ll_score = flan_sample[f'log_likelihood_{aggr}']
                            if ll_score > best_score:
                                best_score = ll_score
                                pred = flan_sample[tgt_key]
                    preds.append(pred)

                model_results[f'self_score_{aggr}'].append(eval_rouge(
                    examples=examples, preds=preds, refs_key=refs_key))

        results[model_name] = \
            {key: np.mean(value) for key, value in model_results.items()}

    print(DataFrame.from_dict(results))


if __name__ == '__main__':
    fire.Fire(main)
