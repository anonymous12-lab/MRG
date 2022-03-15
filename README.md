# Meta-Gen: Update in progress

### Model we use for experiments:

**Pegasus** :[PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)

**Link:** https://github.com/bondarchukb/PEGASUS

------

**Supert:** [SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization](https://aclanthology.org/2020.acl-main.124.pdf)

**Link:** https://github.com/yg211/acl20-ref-free-eval

------

**BertSum:** [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318.pdf)

**Link:** https://github.com/nlpyang/BertSum

------

**TextRank:** [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

**Link:** https://github.com/DerwenAI/pytextrank

------

### Dataset: [sample.csv](https://github.com/anonymous12-lab/MRG/blob/main/sample.csv) is tab-separated file with columns metreview, reviews, recommendation and confidence score along with final decision.

------

**Pre-Processing data in required format for each model: __RUN__**

Pegasus: ```python converttopegasus.py```


SUPERT: ```python converttosupert.py```

------
