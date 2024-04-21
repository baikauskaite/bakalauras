# FLUE: French Language Understanding Evaluation

# Introduction

**FLUE** is an evaluation setup for French NLP systems similar to the popular GLUE benchmark. The goal is to enable further reproducible experiments in the future and to share models and progress on the French language. The tasks and data are obtained from existing works, please refer to our [Flaubert paper](https://arxiv.org/abs/1912.05372) for a complete list of references.

On this page we describe the tasks and provide examples of usage.

In the following, you should replace `$DATA_DIR` with a location on your computer, e.g. `~/data/cls`, `~/data/pawsx`, `~/data/xnli`, etc. depending on the task. Raw data is downloaded and saved to `$DATA_DIR/raw` by running the below command
```bash
bash get-data-${task}.sh $DATA_DIR
```
where `${task}` is either `cls, pawsx, xnli`.

`$MODEL_DIR` denotes the path to where you save the pretrainded FlauBERT model, which contains 3 files:
- `*.pth`: FlauBERT's pretrained model.
- `codes`: BPE codes learned on the training data.
- `vocab`: BPE vocabulary file.

You can download these pretrained models from [here](https://zenodo.org/record/3626826).

# Paraphrasing (PAWS-X)
## Task description
The task consists in identifying whether the two sentences in a pair are semantically equivalent or not.

## Dataset
The train set includes 49.4k examples, the dev and test sets each comprises nearly 2k examples. We take the related datasets for French to perform the paraphrasing task and report the accuracy on the test set.

**Download**:
```bash
bash flue/get-data-pawsx.sh $DATA_DIR
```
The ouput files obtained from the above script are: `$DATA_DIR/raw/x-final/${lang}`, where `${lang}` includes `de, en, es, fr, ja, ko, zh`. Each folder comprises 3 files: `translated_train.tsv`, `dev_2k.tsv`, and `test_2k.tsv`

In this task, we use the related datasets for French (`fr`).

## Example
### Finetuning FlauBERT with Hugging Face's transformers library

#### Preprocess data
Run the below command to prepare data for finetuning. The tokenization (Moses and BPE) is handled later using `FlaubertTokenizer` class in the fine-tuning script.

```bash
python flue/extract_pawsx.py --indir ~/Data/FLUE/pawsx/raw/x-final \
                             --outdir ~/Data/FLUE/pawsx/processed \
                             --use_hugging_face True
```

#### Finetune
<!-- Run the below command to finetune Flaubert using the Transformers repo from [above](#b.-Finetuning-FlauBERT-with-Hugging-Face's-transformers-library), where `~/transformers` should be replaced by the local path where you save the forked repo. -->
Run the below command to finetune Flaubert on `PAWSX` dataset using [Hugging Face's Transformers](https://github.com/huggingface/transformers) library.

```bash
config='flue/examples/pawsx_lr5e6_hf_base_cased.cfg'
source $config

python ~/transformers/examples/run_glue.py \
                                        --data_dir $data_dir \
                                        --model_type flaubert \
                                        --model_name_or_path $model_name_or_path \
                                        --task_name $task_name \
                                        --output_dir $output_dir \
                                        --max_seq_length 512 \
                                        --do_train \
                                        --do_eval \
                                        --learning_rate $lr \
                                        --num_train_epochs $epochs \
                                        --save_steps $save_steps \
                                        --fp16 \
                                        --fp16_opt_level O1 \
                                        |& tee output.log
```

# Citation
If you use FlauBERT or the FLUE Benchmark for your scientific publication, or if you find the resources in this repository useful, please refer to our [paper](https://arxiv.org/abs/1912.05372):

```
@misc{le2019flaubert,
    title={FlauBERT: Unsupervised Language Model Pre-training for French},
    author={Hang Le and Loïc Vial and Jibril Frej and Vincent Segonne and Maximin Coavoux and Benjamin Lecouteux and Alexandre Allauzen and Benoît Crabbé and Laurent Besacier and Didier Schwab},
    year={2019},
    eprint={1912.05372},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
