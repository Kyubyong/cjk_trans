# Pre-trained Machine Translation Models of Korean from/to ECJ

Pre-trained models are beautiful. They save your time, energy and/or money. 
You can obtain several pre-trained machine translation models for mostly European languages [here](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md).
In this project, I add six other models: Korean <-> English, Chinese, Japanese as I failed to find publicly available
 ones.
Not surprisingly, the biggest challenge in training NMT models for those language pairs is the lack of large parallel corpora.
I decided to use both public data ([OpenSubtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php)) and private data) to overcome the difficulties.
Overall, each of their performance may not so impressive, but you can keep training it with your own data, if necessary.

## Requirements
* python >=3.6
* pytoch >=1.0
* [Fairseq](https://github.com/pytorch/fairseq)


## Data
|Language Pair | # Training sents (public + private) | # Test sents (private) |
|--|--|--|--|
|ko-en | 1,845,445 (1,391,190 + 454,255) | 1,050 | 
|ko-zh | 672,450 (485,843 + 186,607) | 1,417 |
| ko-ja | 2,788,003 (302,063 + 2,485,940) | 1,174 |

## Model
* [Transformer Base](https://arxiv.org/abs/1706.03762)

## Vocabulary and tokenization
* Click the links to download the pretrained models and vocabulary files.

|Language | # Vocab. | Tokenization |
|--|--|--|
|[ko](https://www.dropbox.com/s/hn2osffn1onycxa/wiki.ko.model?dl=0) | [8k](https://www.dropbox.com/s/98vmysovz8hpv6x/wiki.ko.dict?dl=0) |  BPE with sentencepiece | 
|[en](https://www.dropbox.com/s/5xoh2sjic1jalbw/gutenberg.model?dl=0) | [32k](https://www.dropbox.com/s/trcrvhd9vs2iwwa/gutenberg.dict?dl=0) | BPE with sentencepiece |
| zh | [32k](https://www.dropbox.com/s/x56g5aqjy7pll51/opensubtitles.zh.dict?dl=0) | character |
| [ja](https://www.dropbox.com/s/37xs58y9hvx9f6f/wiki.ja.model?dl=0) | [8k](https://www.dropbox.com/s/wqk5ba9m2dfbujg/wiki.ja.dict?dl=0) | BPE with sentencepiece |


## Pre-trained models and their performance

|  Pre-trained model | BLEU on test set* | 
|--|--|
|  [ko -> en](https://www.dropbox.com/s/cmvkxxk1zr2cmnf/ko-en.zip?dl=0) | 16.7 |
|  [en -> ko](https://www.dropbox.com/s/t8l9lk61rwiica5/en-ko.zip?dl=0) | 24.2 |
| [ko -> zh](https://www.dropbox.com/s/wp2d05403f5r9xq/ko-zh.zip?dl=0) | 17.13 | 
|[zh -> ko](https://www.dropbox.com/s/qe1q4uslmvkyoa2/zh-ko.zip?dl=0) | 23.78 |
| [ko -> ja](https://www.dropbox.com/s/r00uu48815jx1j1/ko-ja.zip?dl=0) |40.7 |
|[ja -> ko](https://www.dropbox.com/s/4fs14yvdn0tq24u/ja-ko.zip?dl=0)| 34.6 |

* Evaluation is based on the tokenization tools such as [Mecab-ko](https://bitbucket.org/eunjeon/mecab-ko/src/master/) (ko), [NLTK punct](https://www.nltk.org/api/nltk.tokenize.html) (en), [pkuseg](https://github.com/lancopku/pkuseg-python) (zh), and [MeCab](https://github.com/SamuraiT/mecab-python3) (ja).)

## Finetuning Examples

```
echo "ko -> en"
python -m torch.distributed.launch  --nproc_per_node 8 FAIRSEQ/train.py    ko-en-bin --arch transformer       --optimizer adam --lr 0.0005 --label-smoothing 0.1 --dropout 0.3       --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt       --weight-decay 0.0001 --criterion label_smoothed_cross_entropy       --max-epoch 80 --warmup-updates 4000 --warmup-init-lr '1e-07'    --adam-betas '(0.9, 0.98)'   --save-dir train/ko-en/ckpt  --save-interval 1 --restore-file checkpoint77.pt
```