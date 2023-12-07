import fire
import json
import numpy as np


def main(bigbench_data_dir='bigbench_data',
         result_dir='bigbench_cappy_results',
         gen_model_name='flan-t5-xxl'):
    results = {}
    for subset_name in json.load(
            open(f'{bigbench_data_dir}/subset_names.json')):
        results[subset_name] = json.load(
            open(f'{result_dir}/{gen_model_name}/{subset_name}.json'))['rougeL']

    print(json.dumps({gen_model_name: results}, indent=4))
    print(f'Average on Big-Bench: {np.mean(list(results.values()))}')


if __name__ == '__main__':
    fire.Fire(main)
