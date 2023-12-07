import json
import os
import fire


def main(bigbench_data_dir='./bigbench_data', cuda_idx=0):
    for gen_model_size in ['xxl', 'xl', 'large', 'base', 'small']:
        gen_model_name = f'flan-t5-{gen_model_size}'
        for subset_name in json.load(
                open(f'{bigbench_data_dir}/subset_names.json')):
            command = (f'CUDA_VISIBLE_DEVICES={cuda_idx} '
                       f'XLA_PYTHON_CLIENT_MEM_FRACTION=.95 '
                       f'python bigbench_cappy_score.py '
                       f'--bigbench_gen_model {gen_model_name} '
                       f'--bigbench_subset_name {subset_name}')
            print(f'COMMAND: {command}')
            os.system(command)


if __name__ == '__main__':
    fire.Fire(main)


