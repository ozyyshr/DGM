# Dialogue Graph Modeling for Conversational Machine Reading



This is the code for the paper **[Dialogue Graph Modeling for Conversational Machine Reading](https://arxiv.org/abs/2012.14827)**.

**[Here](https://worksheets.codalab.org/bundles/0x3d8355b00e7b44f1b1474a1ddc23f375)** is a codalab bundle link to reproduce our results.

<h3>1. Requirements</h3>

(Our experiment environment for reference)

- Python 3.6
- Python 2.7 (for open discourse tagging tool)

- Pytorch (1.6.0)

- NLTK (3.4.5)

- spacy (2.0.16)
- transformers (2.8.0)
- editdistance (0.5.2)
- dgl (0.5.3)



 <h3>2. Datasets</h3>

[ShARC]: https://sharc-data.github.io/data/sharc1-official.zip

Download the dataset and extract it, or to use it directly in the directory `data/sharc_raw`



<h3>3. Instructions</h3>

<h4>3.1 Preprocess Data</h4>

##### Fixing errors in raw data

```bash
python fix_question.py
```

##### EDU segmentation
The environment requirements are listed **[here](https://www.dropbox.com/sh/tsr4ixfaosk2ecf/AACvXU6gbZfGLatPXDrzNcXCa?dl=0&preview=requirements.txt)**
```bash
cd segedu
python preprocess_discourse_segment.py
python sharc_discourse_segmentation.py
```

##### Discourse relations tagging
We need to train a discourse relation tagging model according to **[here](https://github.com/shizhouxing/DialogueDiscourseParsing)**. 
Firstly, download Glove for pretrained word vector and put it in `DialogueDiscourseParsing/glove/glove.6B.100d.txt`.

Secondly, preprocess data for training.

```bash
python data_pre.py <input_dir> <output_file>
```

Or you can directly use the data in `DialogueDiscourseParsing/data/processed_data`.

Then train the parser with 

```bash
python main.py --train
```

The model should be stored in `DialogueDiscourseParsing/dev_model`.
One can directly use the model trained **[here](https://drive.google.com/file/d/1NsxUjapp-iynWAwUxGmyk1EmI7YlRIZq/view?usp=sharing)**.

Finally, we can inference for ShARC dataset to get the discourse relations. 

```bash
python construct_tree_mapping.py
python convert.py

cd DialogueDiscourseParsing
python main_.py
```

##### Preprocessing for Decision Making

```
python preprocess_decision_base.py
```

##### Preprocessing for Question Generation

```
python preprocess_span.py
```

All the preprocessed data can be found in the directory `./data`. You can also download it **[here](https://drive.google.com/drive/folders/1QepEf4Uu3GHCsF1L7TuM5uSADlcex-7v?usp=sharing)**

<h4>3.2 Decision Making and Question Generation</h4>

To train the model on decision making subtask, run the following:

```bash
python -u train_sharc.py \
--train_batch=16 \
--gradient_accumulation_steps=2 \
--epoch=5 \
--seed=323 \
--learning_rate=5e-5 \
--loss_entail_weight=3.0 \
--dsave="out/{}" \
--model=decision_gcn \
--early_stop=dev_0a_combined \
--data=./data/ \
--data_type=decision_electra-large-discriminator \
--prefix=train_decision \
--trans_layer=2 \
--eval_every_steps=300
```

The trained model and corresponding results are stored in `out/train_decision`

For question generation subtask, we first extract the under-specified span by following:

```bash
python -u train_sharc.py \
--train_batch=16 \
--gradient_accumulation_steps=2 \
--epoch=5 \
--seed=115 \
--learning_rate=5e-5 \
--dsave="out/{}" \
--model=span \
--early_stop=dev_0_combined \
--data=./data/ \
--data_type=span_electra-large-discriminator \
--prefix=train_span \
--eval_every_steps=100
```

The trained model and corresponding results are stored in `out/train_span`

Then, use the **inference result** of under-specified span and the rule document to generate follow-up questions:

```bash
python -u qg.py \
--fin=./data/sharc_raw/json/sharc_dev.json \
--fpred=./out/inference_span \  # directory of span prediction
--model_recover_path=/absolute/path/to/pretrained_models/qg.bin \
--cache_path=/absolute/path/to/pretrain_models/unilm/
```

The final results are stored in `final_res.json`

<h3>Acknowledgement</h3>
Part of code is modified from the **[Discern](https://github.com/Yifan-Gao/Discern) implementation**.
