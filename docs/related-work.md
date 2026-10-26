# Related work

- A. Ploshkin et al., **Yambda-5B: A Large-Scale Multi-modal Dataset for Ranking and Retrieval**, 2025. [Primary paper](https://arxiv.org/abs/2505.22238). It introduces a music-interaction resource and global temporal evaluation. Our pinned 50m release, retained cohort and sampled-positive ranking task are explicitly narrower than reproducing its full benchmark.
- J. M. Kanter and K. Veeramachaneni, **Deep Feature Synthesis: Towards Automating Data Science Endeavors**, IEEE DSAA, 2015. [Authors' paper](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf). It follows relational paths and composes transformations to generate features. Our grammar restricts that broad idea to declared user/item/pair temporal counts, decays and ratios.
- U. Khurana, H. Samulowitz and D. Turaga, **Feature Engineering for Predictive Modeling Using Reinforcement Learning**, AAAI, 2018. [Publisher page](https://ojs.aaai.org/index.php/AAAI/article/view/11678). It explores a transformation graph with a learned strategy. We implement no reinforcement-learning policy: our one-step enumeration exposes exactly the candidate groups and fit budget.

No claim is made to reproduce those methods' reported benchmark numbers, or that our small search is a novel optimizer.
