## Adversarial Training in MRC

---

### Adversarial Examples for Evaluating Reading Comprehension Systems

论文小结: create adversarial examples by adding distracting sentences to the input paragraph,

We automatically generate these sentences so that they confuse models, but do not contradict the correct answer or confuse humans.

### A Robust Adversarial Training Approach to Machine Reading Comprehension

论文小结: Specificly, dynamically generates adversarial examples based on the parameters of current model and further trains the model by using the generated examples in an iterative schedule. it does not require any specification of adversarial attack types

### Evaluating Neural Machine Comprehension Model Robustness to Noisy Inputs and Adversarial Attacks

### Beat the AI: Investigating Adversarial Human Annotation for Reading Comprehension

论文小结: 通过提供SQuAD1.1种的文章，以及认为给定合理的问题，找到当前3种模型无法回答的样本。

### Adversarial NLI: A New Benchmark for Natural Language Understanding