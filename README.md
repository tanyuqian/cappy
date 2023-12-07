# Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer

This repo contains code of the following paper:

**Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer** \
Bowen Tan, Yun Zhu, Lijuan Liu, Eric Xing, Zhiting Hu, Jindong Chen \
NeurIPS 2023 \
[[arXiv]](https://arxiv.org/pdf/2311.06720.pdf)  [[Model Card (btan2/cappy-large)]](https://huggingface.co/btan2/cappy-large)


## Getting Started

* Cappy is a pretrained small scorer designed to enhance the performance and efficiency of multi-task LLMs. 
* Cappy takes in an instruction and a candidate response as input, and produces a score between 0 and 1, indicating an estimated correctness of the response with respect to the instruction. 
* With merely 360 million parameters, Cappy functions either independently on classification tasks or serve as an auxiliary component for LLMs, boosting their performance. 
* Also, Cappy enables efficiently integrating downstream supervision without requiring LLM finetuning nor the access to their parameters.
* Furthermore, Cappy is flexible to cooperate with other LLM adaptations, including finetuning and in-context learning, and prompt tuning, offering additional performance enhancement.

 

Now, Cappy can be loaded with `transformers` either as a Jax/Flax model or a PyTorch model.

### Jax/Flax
```python
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('btan2/cappy-large')
cappy = FlaxAutoModelForSequenceClassification.from_pretrained('btan2/cappy-large')

instruction = """
What label best describes this news article?
Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group,\which has a reputation for making well-timed and occasionally\controversial plays in the defense industry, has quietly placed\its bets on another part of the market.
"""
response = 'Business'

inputs = tokenizer([(instruction, response), ], return_tensors='pt')
score = cappy(**inputs).logits[0][0].item()
```

### PyTorch
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('btan2/cappy-large')
cappy = AutoModelForSequenceClassification.from_pretrained('btan2/cappy-large')

instruction = """
What label best describes this news article?
Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group,\which has a reputation for making well-timed and occasionally\controversial plays in the defense industry, has quietly placed\its bets on another part of the market.
"""
response = 'Business'

inputs = tokenizer([(instruction, response), ], return_tensors='pt')
score = cappy(**inputs).logits[0][0].item()
```



Below are the scripts to recover the experiments in the paper.

## Requirements

Cappy's pretraining and finetuning are both based on [Redco](https://github.com/tanyuqian/redco), 
a lightweight tool automating distributed training on both GPUs and TPUs. 

To install redco
```shell
pip install redco==0.4.13
```

Sometimes the Jax version needs be adjusted based on your device & environment. 
Here are some [instructions](https://github.com/tanyuqian/redco#adjust-jax--flax-versions).

To install other requirements,
```shell
pip install -r requirements.txt
```


## Pretraining Cappy

Cappy's pretraining uses the code from [this example](https://github.com/tanyuqian/redco/tree/master/examples/classification_regression) in Redco. We will release Cappy's pretraining data soon.


## Evaluting Cappy on PromptSource (zero-shot)

### Download Test Data
Following the setting from [OPT-IML paper](https://arxiv.org/pdf/2212.12017.pdf) (Section 5.2). We conduct zero-shot evaluation on 11 held-out classification tasks from PromptSource.  
```shell
bash scripts/download_promptsource_test_data.sh
```

### Running Cappy
```shell
python cappy_promptsource.py --model_name_or_path btan2/cappy-large
```

### Results

|             | OPT 30B | OPT-IML 30B | OPT 175B | OPT-IML 175B | T0 11B | Cappy (ours, 0.36B) |
|------------:|:-------:|:-----------:|:--------:|:------------:|:------:|:-------------------:|
|     ANLI R1 |   33.7  |     37.1    |   34.1   |     42.2     |  42.1  |        34.3         |
|     ANLI R2 |   34.1  |     35.4    |   34.1   |     38.5     |  37.9  |        33.9         |
|     ANLI R3 |   34.7  |     36.6    |   34.7   |     39.6     |  39.7  |        34.7         |
|          CB |   24.6  |     43.2    |   38.9   |     56.4     |  58.5  |        59.4         |
|         RTE |   56.4  |     67.8    |   54.0   |     73.4     |  80.2  |        71.9         |
|  StoryCloze |   55.5  |     90.7    |   57.0   |     95.0     |  96.7  |        93.7         |
|         WSC |   43.5  |     58.2    |   51.0   |     59.2     |  58.6  |        63.8         |
|         WiC |   50.8  |     54.7    |   49.7   |     53.6     |  56.0  |        51.9         |
|  Winogrande |   50.2  |     53.4    |   50.1   |     56.6     |  62.5  |        51.7         |
|  WinoGender |   54.9  |     64.6    |   53.9   |     72.7     |  83.8  |        68.9         |
| Crows-Pairs |   85.5  |     22.3    |   85.5   |     34.4     |  24.0  |        57.8         |
| **Average** |   47.6  |     51.3    |   49.3   |     56.5     |  58.2  |        56.6         |

Baseline results come from [OPT-IML paper](https://arxiv.org/pdf/2212.12017.pdf) (Section 5.2).


## Boosting FLAN-T5 with Cappy on Big-Bench Tasks

### Getting Big-Bench Tasks

We take all 45 generative tasks from Big-Bench in our experiment. The command below process the tasks into `.jsonl` format.

```shell
python scripts/get_bigbench_data.py
```
The processed datasets can be found in `./bigbench_data`, where `./bigbench_data/subset_names.json` records all the task names.


### Getting FLAN-T5 Outputs

We collect generated outputs (as well as log-likelihoods on evaluation sets) from FLAN-T5 models (from `-small` to `-xxl`). They can be downloaded with
```shell
bash scripts/download_bigbench_flan_gens.sh
```

If you want to generate outputs by your self and/or adjust some generation settings, we provide generation code as below that supports distributed inference using multiple GPUs together (in case the model is too large to accomodate on a single GPU, e.g., `FLAN-T5-XXL (11B)`).
```shell
python scripts/bigbench_flan_generate.py \
  --model_name_or_path google/flan-t5-xl \
  --n_model_shards 4
```
where `--n_model_shards` refers to the number of shards you want to split the large model into (it's usually the number of GPUs on your device if it's not 1).

### Adapting Cappy to boost FLAN-T5

```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python cappy_bigbench.py \
  --model_name_or_path btan2/cappy-large \
  --bigbench_subset_name auto_categorization \
  --bigbench_gen_model flan-t5-xxl \
  --train_size 102400
```
* `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`: (In case GPU memory exceeds) adjust the GPU memory pre-allocation to Jax, see [here](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) for more details. 
* `--bigbench_subset_name`: the name of subset from Big-Bench (see `./bigbench_data/subset_names.json` for all of them).
* `--bigbench_gen_model`: the FLAN model to be boosted. 
* `--train_size`: the target data size to construct for Cappy's finetuning on the task (collect FLAN outputs, and then truncate or repeat). 

See `def main(...)` in [cappy_bigbench.py](cappy_bigbench.py) for all the arguments.

Every sub-task takes 40 mins to run on a single A10G GPU. The result will be logged in `./bigbench_cappy_results/{flan_model}/{subset_name}.json`.


Besides, to run all the Big-Bench subsets at once, 
```shell
python scripts/run_cappy_bigbench.py --cuda_idx 0
```

### Results

To present baseline results, `python scripts/present_bigbench_baselines.py`

To present Cappy results on all 45 Big-Bench subtasks,
`python scripts/present_cappy_bigbench_results.py --gen_model_name flan-t5-xxl`

The reported numbers on the paper are produced on TPU machines. Here we provide our
reproduction results on A10G GPUs in `./bigbench_cappy_results`. The gap between
them is slight (`ΔrougeL <= 0.8`).

|                      | flan-t5-small | flan-t5-base | flan-t5-large | flan-t5-xl  | flan-t5-xxl |
|----------------------|---------------|--------------|---------------|-------------|-------------|
| Beam Search (beam=4) | 16.4025       | 19.8594      | 23.4802       | 26.1177     | 29.6608     |
| Sampling             | 11.4317       | 15.7909      | 19.6248       | 23.2191     | 25.7273     |
| Temperature (t=0.9)  | 12.0126       | 17.0571      | 20.0481       | 24.2702     | 27.0985     |
| Topk (k=40)          | 11.5157       | 15.7481      | 19.7634       | 22.6692     | 25.8226     |
| Nucleus (p=0.95)     | 11.9171       | 16.6174      | 20.1986       | 24.1654     | 26.9036     |
| Self-Score (sum)     | 15.0806       | 20.711       | 24.1224       | 28.4665     | 32.0156     |
| Self-Score (mean)    | 16.4223       | 20.1317      | 23.7828       | 26.7694     | 30.246      |
| **Cappy (ours)**     | **23.6543**   | **27.6178**  | **30.3802**   | **33.2775** | **37.1678** |

## Acknowledgement

*Cappy* is Mario's ally throughout Super Mario Odyssey and assists him in various ways. We thank Nintendo for the nice game!

![](imgs/cappy.jpg)