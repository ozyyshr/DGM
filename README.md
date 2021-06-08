# Dialogue Graph Modeling for Conversational Machine Reading

This is the code for the paper **[Dialogue Graph Modeling for Conversational Machine Reading](https://arxiv.org/abs/2012.14827)**.



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

```bash
cd segedu
python preprocess_discourse_segment.py
python sharc_discourse_segmentation.py
```

##### Discourse relations tagging

```bash
cd DialogueDiscourseParsing
python main_.py
```

Note that we didn't put the pre-trained model in the code package due to limit size. 

##### Preprocessing for Decision Making

```
python preprocess_decision_base.py
```

##### Preprocessing for Question Generation

```
python preprocess_span.py
```

All the preprocessed data can be found in the directory `./data`

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
--model=decision \
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
