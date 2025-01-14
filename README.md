# sequential-fterm-classifier
A patent classifier that sequentially predicts f-terms for patent abtracts with a wide range of applications in practice and research. Technological vector representations are provided by the model and can be used for various metrics and applications.

The model is available [here](https://huggingface.co/RWTH-TIME/galactica-125m-f-term-classification).

The paper discussing the training, limitations and potential use-cases of the model can be found [here](https://aisel.aisnet.org/icis2024/data_soc/data_soc/3/).

As of now we are working on a significantly advanced model that will be released in future. 


## Model Information
This model is based on the Facebook Galactica-125m model, retrained for classifying patent abstracts under the F-term classification system. The base model, facebook/galactica-125m, was extended with 378,165 tokens, each corresponding to unique F-terms. These terms represent granular technical attributes of patents. A new, randomly initialized classification head replaced the original, enabling multi-label classification exclusively for F-terms without generating ordinary text tokens.

The primary purpose of this model is to address limitations of traditional hierarchical patent classification systems (e.g., IPC, CPC) by enabling:

- Granular and **horizontal comparison of patents** within and across technological domains.
- **Cross-domain analyses** using vectorized representations of F-terms.
- **Consistent global classification** to improve comparability across patent offices.


|                       | Training Data                                                  | Params | Input Modalities  | Output Modalities         | Vocabulary Size |
| :-------------------- | :------------------------------------------------------------ | :----: | :---------------: | :-----------------------: | :-------------: |
| F-term Classifier     | 7,478,671 patent abstracts with F-term classifications from EPO Patstat | 670M  | Patent abstracts | F-term classifications (vector and text) | 428,165         |


## Training
The model was retrained on a preprocessed dataset derived from the EPO Patstat database, containing 7,478,671 English-language patent abstracts and their associated F-term classifications. Each patent was tagged with multiple F-terms, describing its technological properties. Key training highlights include:

- **Data Augmentation**: Shuffling the order of F-terms during training to discourage reliance on sequential patterns.
- **Hardware**: Training leveraged an NVIDIA RTX 4090 GPU over 3 epochs.
- **Performance**: Achieved a top-1 precision of 42.68% and a top-5 precision/success of 61.14% for predicting correct F-terms.


## Vector Representations
The model provides vectorized embeddings for F-terms, enabling:

- Metric-based comparisons (e.g., cosine similarity) of technological attributes.
- Analysis of cross-domain and interdisciplinary technological innovation.
- Enhanced patent-based metrics, such as technological distance and diversity.
These embeddings are derived from the weights of the classification head and have been validated using dimensionality reduction techniques like t-SNE, confirming meaningful clustering of related F-terms.

## Use Cases
- **Patent Analysis**: Enables detailed exploration of technological attributes and cross-domain innovation.
- **Firm and Competitor Analysis**: Facilitates more accurate mapping of technological portfolios.
- **Policy and Strategic Planning**: Supports unbiased, global patent analysis.
- **Cross-Domain Technology Research**: Breaks down silos inherent in hierarchical classification systems.
- **Technology Opportunity Discovery**: Identifies emerging opportunities by analyzing vectors to uncover novel connections between disparate technological domains or attributes, enabling strategic foresight.

## Limitations
- Classification Challenges: Errors primarily occur at granular term levels within themes, highlighting room for improvement in differentiating subtle attributes. We are actively working on improving the models performance.
- F-term Bias: Since F-terms originate from Japanese patents, potential biases from the JPO's classification practices may influence predictions.

## Recommended Citation
Selzner, Paul; Beckers, Lukas; Dienhart, Christina; and Antons, David, "Addressing Limitations of Patent Research Using Machine-Learning: A Research Agenda Based on Automatic F-term Classification and Technology Spanning Vector Data" (2024). ICIS 2024 Proceedings. 3.
https://aisel.aisnet.org/icis2024/data_soc/data_soc/3
