# 🌡️ Climate Impact Deep Learning Models

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Functional%20API-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)
![Colab](https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)

**[Open in Colab](https://colab.research.google.com/github/StanStarishko/climate-impact-ml/blob/main/dl_machine_learning_assessment.ipynb)**

> GitHub does not render this notebook reliably because of its size and embedded outputs. The intended way to view and run it is in Google Colab using the link above.

## The Story

This project started as a college final assessment, with a brief written like a real consulting engagement: a climate organisation hires me for two machine learning projects, Arctic Sea Ice forecasting and Forest Fire detection. I had two options. Treat it as homework and tick the boxes. Or treat it like an actual job, build something I would not be embarrassed to ship, and write about it like a real engineer would.

I picked the second option, and this is what came out of it.

## What I Built

Two deep learning models that share one architecture and solve completely different problems.

**The Sea Ice model** is a regression that predicts tomorrow's Arctic sea ice extent from the previous ten days of weather. Mean absolute percentage error landed at **1.27%** on the held-out test set, which is small enough to be operationally useful and was the basis for an unconditional deploy recommendation in the writeup.

**The Forest Fire model** is a binary classifier on 8x8 RGB images, trying to spot fire in heavily imbalanced data (1 fire image per 80 not-fire). Headline accuracy is 99.12%, but that number is mostly the trivial baseline. The actual signal is **F1 score of 0.6422 on the fire class**, achieved by class weighting in the loss function, stratified train/test split, and a threshold-tuning analysis that informed a conditional deploy recommendation: useful as a screening layer in a multi-stage pipeline, not as a standalone fire detector.

Both models are built with the Keras Functional API on the same backbone: input projection, two inverted bottleneck blocks with residual connections, GELU activations, layer normalisation, dropout. About 285 thousand parameters each. The only architectural difference is the output head, one scalar with GELU activation for regression, two units with softmax for classification. Same idea, two tasks.

## What I Learned

The template I was given had two non-obvious bugs. The first was data leakage in the Sea Ice pipeline: standardisation statistics were computed on the full dataset before splitting train and test, which leaked test distribution information into the scaler. I caught it, commented out the original code with a first-person justification of what was wrong and why, and replaced it with a fit-on-train-only version. The second bug was a naive slice-based split for Forest Fire that, combined with a 1:80 class imbalance and lexicographically sorted file names, could have landed almost no fire examples in the test set. Stratified split fixed that. Both finds came from reading the template critically rather than copying it. Reading code carefully is a real engineering skill, and I now believe it is worth more hours than people give it credit for.

The class imbalance work on Forest Fire was the most interesting part. A naive 0.5-second EDA check revealed the 1:80 ratio, which then drove every subsequent decision: class weighting in `model.fit`, stratified split, threshold tuning, classification report instead of accuracy as the headline metric, deploy decision tied to operational context rather than to a single number. I spent more time thinking about what the metrics meant than I spent training the model. This is the right ratio for real ML work, even if it does not feel like it when you start.

Threshold tuning was the surprise. I expected the standard sharp precision-recall trade-off curve, with F1 peaking somewhere off the default 0.5. What I got was nearly flat curves across the entire 0.05 to 0.95 range, which told me the model produced confident outputs and that further gains would not come from tuning thresholds but from better data or a different architecture (CNN at higher resolution, probably). Sometimes the most important finding in an analysis is that the lever you were going to pull does not actually move much.

## How I Work

I treat ML projects like any other engineering work. Plan the pipeline before writing code. Document decisions in plain language inline with the implementation. When I find a bug, I do not silently rewrite the surrounding code. I comment out the original, explain what is wrong, and write the fix with reasoning. Future me, future colleague, future assessor: all of them deserve to see the thinking, not just the result.

The notebook is structured so each practical question is answered in code, then followed by a markdown deploy decision that argues for or against shipping the model with specific operational conditions. The two deploy decisions read very differently on purpose. Sea Ice is an unconditional deploy with light monitoring. Forest Fire is a conditional deploy as a screening layer in a multi-stage pipeline, with a recall-favourable threshold and continuous monitoring. The contrast is the lesson: deploy decisions are not "good model" versus "bad model", they are "what role can this model safely play in a real system".

## Repository Structure

```
climate-impact-ml/
├── README.md                              you are reading it
├── dl_machine_learning_assessment.ipynb   the project notebook
└── datasets/
    ├── sea_ice_dataset.csv                Sea Ice daily weather records (11K+ rows)
    ├── forest_fire_dataset.zip            Forest Fire images (part 1 of multi-part archive)
    └── forest_fire_dataset.z01            Forest Fire images (part 2 of multi-part archive)
```

The notebook downloads the datasets directly from this repository at runtime via raw GitHub URLs, so there is no manual setup required.

## Running It

The fastest way is to open the notebook in Colab and run all cells:

**[Open in Colab](https://colab.research.google.com/github/StanStarishko/climate-impact-ml/blob/main/dl_machine_learning_assessment.ipynb)**

No local setup, no data files to download manually. Total runtime is about 10 to 15 minutes, most of which is Forest Fire training on roughly 60 thousand images.

## From Heart Attack Risk Factors to Deep Learning

This is the next step from my [Heart Attack Risk Factors](https://github.com/StanStarishko/Portfolio/tree/main/Python/Heart%20Attack%20Risk%20Factors) project from August 2024. Where that one was about asking pandas the right questions, this one is about teaching neural networks to answer them. Same family of skills (Python, data wrangling, visualisation, getting insights from real data), turned up several levels: end-to-end deep learning pipelines, two production-style models, imbalanced classification, threshold analysis, and deploy decisions written for actual operational use rather than a tick-box submission.

If you want to see the progression from "make pretty charts" to "ship a model with conditions", these two projects are the bookends.

---

**© May 2026 Stanislav Starishko**
