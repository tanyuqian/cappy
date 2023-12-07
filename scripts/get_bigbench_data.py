import os
import fire
import json
import tqdm
from datasets import get_dataset_config_names, load_dataset


def main(cache_dir='./cache', output_dir='./bigbench_data'):
    os.makedirs(output_dir, exist_ok=True)

    subset_names = []
    for subset_name in tqdm.tqdm(
            get_dataset_config_names('tasksource/bigbench'),
            desc='processing BigBench datasets'):
        if subset_name == 'simple_arithmetic_json_subtasks':
            continue

        try:
            ds = load_dataset(
                'tasksource/bigbench', subset_name, cache_dir=cache_dir)
        except:
            print(f'subset {subset_name} not loadable.')
            continue

        if len(ds['train'][0]['multiple_choice_targets']) > 0:
            continue
        else:
            subset_names.append(subset_name)

        for split in ['train', 'validation']:
            with (open(f'{output_dir}/{subset_name}_{split}.jsonl', 'w') as
                  output_file):
                for example_idx, example in enumerate(ds[split]):
                    output_file.write(json.dumps({
                        'dataset': subset_name,
                        'example_idx': example_idx,
                        'instruction': example['inputs'],
                        'references': example['targets']
                    }) + '\n')

    assert len(subset_names) == 45
    json.dump(
        subset_names, open(f'{output_dir}/subset_names.json', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)