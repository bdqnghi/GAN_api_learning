## Domain Adaptation for Cross Language API Learning 
![Model](figs/self_refinement.png)

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch](http://pytorch.org/)
* [Faiss](https://github.com/facebookresearch/faiss) (recommended) for fast nearest neighbor search (CPU or GPU).

Available on CPU or GPU, in Python 2 or 3. Faiss is *optional* for GPU users - though Faiss-GPU will greatly speed up nearest neighbor search - and *highly recommended* for CPU users. Faiss can be installed using "conda install faiss-cpu -c pytorch" or "conda install faiss-gpu -c pytorch".

## Align monolingual word embeddings

* **Unsupervised**: learn a mapping from the source to the target space using adversarial training and (iterative) Procrustes refinement.

For more details on these approaches, please check [here](https://arxiv.org/pdf/1710.04087.pdf).

### The unsupervised way: adversarial training and refinement (CPU|GPU)
A sample command to learn a mapping using adversarial training and iterative Procrustes refinement:
```bash
python3 unsupervised.py --src_lang java --tgt_lang cs --src_emb data/java_vectors_sdk_functions_api_tokens_with_keywords_50_15.txt --tgt_emb data/cs_vectors_sdk_functions_api_tokens_with_keywords_50_15.txt --n_refinement 2 --emb_dim 50 --max_vocab 300000 --epoch_size 100000 --n_epochs 1 --identical_dict_path "dict/candidates_dict.txt" --dico_eval "eval/java-cs.txt"
```
### Evaluate cross-lingual embeddings (CPU|GPU)

```bash
python evaluate.py --src_lang java --tgt_lang cs --src_emb dumped/debug/some_id/vectors-java.txt --tgt_emb dumped/debug/some_id/vectors-cs.txt --dico_eval "eval/java-cs.txt" --max_vocab 200000
```
### Some explanations and tips:
* **n_epochs**: number of epoch, usually up to 5 is good enough.
* **epoch_size**: size of the epoch to run over the training data once, for large vocabulary(e.g 100.000 words), should be around 500.000-1.000.000. Current default is 100.000
* **n_refinement**: number of refinement steps, usually the result converges after 2 iterations if the initial results is already good.
* **emb_dim**: size of the input embeddings, now default is 50, 50 is also the recommended size to get a good performance. 
* **identical_dict_path**: path to the synthetic dictionary. Since we're based on class and method name to induce a synthetic dictionary for the refinement, it need to be precalculated and store to somewhere first, otherwise the computation will be slow if the size of the 2 input embeddings is large.
* **dico_eval**: path to the evaluation dictionary
* If the discriminator loss reaches 0.35, it's a good sign that the model converges, more training may not affect much.
* After the training step, a new folder is generated under "dumped/debug" with a unique ID each time a script is running, new embeddings are writtent in there.
