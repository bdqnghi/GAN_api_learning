## MUSE: Multilingual Unsupervised and Supervised Embeddings
![Model](https://s3.amazonaws.com/arrival/outline_all.png)

MUSE is a Python library for *multilingual word embeddings*, whose goal is to provide the community with:
* state-of-the-art multilingual word embeddings ([fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) embeddings aligned in a common space)
* large-scale high-quality bilingual dictionaries for training and evaluation

We include two methods, one *supervised* that uses a bilingual dictionary or identical character strings, and one *unsupervised* that does not use any parallel data (see [Word Translation without Parallel Data](https://arxiv.org/pdf/1710.04087.pdf) for more details).

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch](http://pytorch.org/)
* [Faiss](https://github.com/facebookresearch/faiss) (recommended) for fast nearest neighbor search (CPU or GPU).

MUSE is available on CPU or GPU, in Python 2 or 3. Faiss is *optional* for GPU users - though Faiss-GPU will greatly speed up nearest neighbor search - and *highly recommended* for CPU users. Faiss can be installed using "conda install faiss-cpu -c pytorch" or "conda install faiss-gpu -c pytorch".

## Get evaluation datasets
Get monolingual and cross-lingual word embeddings evaluation datasets:
* Our 110 [bilingual dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries)
* 28 monolingual word similarity tasks for 6 languages, and the English word analogy task
* Cross-lingual word similarity tasks from [SemEval2017](http://alt.qcri.org/semeval2017/task2/)
* Sentence translation retrieval with [Europarl](http://www.statmt.org/europarl/) corpora

by simply running (in data/):

```bash
./get_evaluation.sh
```
*Note: Requires bash 4. The download of Europarl is disabled by default (slow), you can enable it [here](https://github.com/facebookresearch/MUSE/blob/master/data/get_evaluation.sh#L99-L100).*

## Get monolingual word embeddings
For pre-trained monolingual word embeddings, we highly recommend [fastText Wikipedia embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md), or using [fastText](https://github.com/facebookresearch/fastText) to train your own word embeddings from your corpus.

You can download the English (en) and Spanish (es) embeddings this way:
```bash
# English fastText Wikipedia embeddings
curl -Lo data/wiki.en.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
# Spanish fastText Wikipedia embeddings
curl -Lo data/wiki.es.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec
```

## Align monolingual word embeddings
This project includes two ways to obtain cross-lingual word embeddings:
* **Supervised**: using a train bilingual dictionary (or identical character strings as anchor points), learn a mapping from the source to the target space using (iterative) [Procrustes](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) alignment.
* **Unsupervised**: without any parallel data or anchor point, learn a mapping from the source to the target space using adversarial training and (iterative) Procrustes refinement.

For more details on these approaches, please check [here](https://arxiv.org/pdf/1710.04087.pdf).

### The supervised way: iterative Procrustes (CPU|GPU)
To learn a mapping between the source and the target space, simply run:
```bash
python supervised.py --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec --n_refinement 5 --dico_train default
```
By default, *dico_train* will point to our ground-truth dictionaries (downloaded above); when set to "identical_char" it will use identical character strings between source and target languages to form a vocabulary. Logs and embeddings will be saved in the dumped/ directory.

### The unsupervised way: adversarial training and refinement (CPU|GPU)
To learn a mapping using adversarial training and iterative Procrustes refinement, run:
```bash
python unsupervised.py --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec --n_refinement 5
```
By default, the validation metric is the mean cosine of word pairs from a synthetic dictionary built with CSLS (Cross-domain similarity local scaling). For some language pairs (e.g. En-Zh),
we recommend to center the embeddings using `--normalize_embeddings center`.

### Evaluate monolingual or cross-lingual embeddings (CPU|GPU)
We also include a simple script to evaluate the quality of monolingual or cross-lingual word embeddings on several tasks:

**Monolingual**
```bash
python evaluate.py --src_lang en --src_emb data/wiki.en.vec --max_vocab 200000
```

**Cross-lingual**
```bash
python evaluate.py --src_lang en --tgt_lang es --src_emb data/wiki.en-es.en.vec --tgt_emb data/wiki.en-es.es.vec --max_vocab 200000
```

