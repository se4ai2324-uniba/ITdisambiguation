IT Disambiguation
==============================

This model was developed for the **Natural Language Processing** and **Computer Vision** courses,
it is able to understand the correct sense of the target word by using a text disambiguation module based on MPNet and WordNet.
MPNet is used to encode text plus some context and then, after getting from WordNet all the synsets associated to the target word, a score is computed between all the encoded word-synset pairs in order to select the correct synset.
The WordNet synset description is then used to expand the word's context and is given as input to a fine tuned CLIP-based model used to connect text to images.

SemEval-2023
------------

The model was specifically built to address the [SemEval-2023 Visual Word Sense Disambiguation](https://raganato.github.io/vwsd/) challenge.

**Task**: Given a word and some limited textual context, the task is to select among a set of candidate images the one which corresponds to the intended meaning of the target word.
```
@inproceedings{raganato-etal-2023-semeval,
    title = "{S}em{E}val-2023 {T}ask 1: {V}isual {W}ord {S}ense {D}isambiguation",
    author = "Raganato, Alessandro  and
      Calixto, Iacer and
      Ushio, Asahi and
      Camacho-Collados, Jose  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
}
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

# Grafana
We used the Grafana tool to be able to graphically display some values ​​obtained from Prometheus metrics by performing queries.
The dashboard is divided into four rows, each displaying data related to the past month:

The first row contains four counters:
- The first counter represents the total number of HTTP requests made.
- The second counter displays the number of responses that returned error status codes (in the 400 range) for GET requests.
- The third counter shows the number of responses returning error status codes (in the 500 range) for GET requests.
- The fourth counter indicates the number of responses returning error status codes (in the 400 range) for POST requests.
- The fifth counter represents the number of responses returning error status codes (in the 500 range) for POST requests.
![Counters Row](./docs/images/Grafana/grafana1.png)

The second row features two time series:
- One related to request sizes measured in bytes.
- The other concerning response sizes, also measured in bytes.
![Size Row](./docs/images/Grafana/grafana2.png)

The third row contains two graphs:
- One displaying the average latency in responses.
- The other highlighting the average latency per handler.
![Latency Row](./docs/images/Grafana/grafana3.png)

In the fourth row, there are two graphs depicting resource usage:
- The first graph illustrates the usage of machine RAM.
- The second graph focuses on the average CPU usage.
![Usage Row](./docs/images/Grafana/grafana4.png)


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
