# SciTweets

<!-- Short Introduction what this repo is about -->
This repository contains a dataset, annotation framework and code for the work *"SciTweets - A Dataset and Annotation Framework for Detecting Scientific Online Discourse"* published at **CIKM2022**. You can also download the dataset and annotation framework from https://doi.org/10.7802/2434.


<!-- *TODO*: refer to our work before being published (e.g arxiv preprint)

*TODO 2*: give examples of scientific online discourse here, e.g "The **SciTweets** dataset consists of ..." -->

__Table of contents:__
- [Contents of the Repository](#contents-of-the-repository)
  - [Directory Structure](#directory-structure)
  - [Statistics](#statistics)
- [Pretrained Model](#pretrained-model)
- [Publication](#publication)
- [Licensing](#licensing)
- [Contact](#credits)
- [Acknowledgment](#acknowledgment)

## Contents of the Repository

### Directory Structure
=======================<br/>
This repository contains the following directories and files:

1. **annotations**
   1. **annotations.tsv** the annotated SciTweets dataset
2. **classifier_preds** the classifier predictions used for evaluation in section 3 of the paper
   1. **scibert_2stage_predictions.tsv** the classifier predictions to compute the prec@100 scores
   2. **scibert_cv_predictions.tsv** the classifier predictions to compute precision, recall and F1
3. **heuristics** code and supplementary material for the heuristics explained in section 2
   1. **cat1_sciknowledge.py** code to run heuristics for category 1.1
   2. **cat2_sciurl.py** code to run heuristics for category 1.2
   3. **cat3_research.py** code to run heuristics for category 1.3
   4. **news_outlets_domains.csv** domains and subdomains from major news outlets
   5. **README.md** information about the heuristics
   6. **repo_subdomains.csv** domains and subdomains from open access repositories 
   7. **sc_methods.txt** list of scientific methods from SAGE Social Science Thesaurus [Link](https://concepts.sagepub.com/vocabularies/social-science/en/page/?uri=https%3A%2F%2Fconcepts.sagepub.com%2Fsocial-science%2Fconcept%2Fconceptgroup%2Fmethods)
   8. **science_mags_domains.csv** domains and subdomains from science magazines
   9. **wiki_sci_terms.txt** list of scientific terms extracted from wikipedia glossaries

4. **annotation_framework.pdf** the annotation framework provided to the annotators
5. **classifier_cv.py** code to reproduce the 10-fold cv results in section 3, table 3
6. **classifier_p100.py** code to reproduce the prec@100 results in section 3, table 4
7. **classifier_eval.ipynb** code to calculate the performance for section 3, table 3 and 4
8. **dataset_statistics.ipynb** code to calculate the dataset statistics for section 2, table 2 
9. **Readme.md** this file

### Statistics
============<br/>
Label distribution of the annotated dataset in annotations/annotations.tsv:

| Labels        |   Category 1   | Category 1.1 |   Category 1.2 | Category 1.3 | 
|---------------|:--------------:|-------------:|---------------:|-------------:|
| Yes           | 403 (31.88%)   | 283 (23.82%) |   190 (15.65%) | 259 (21.32%) |
| No            |  859 (68.12%)  | 905 (76.18%) | 1024 (84.35%)  | 956 (78.68%) | 

More statistics available in **dataset_statistics.ipynb**.

More information about the categories available in **annotation_framework.pdf**

## Pretrained Model
The SciBert baseline classifier trained in section 3 of the paper can be used for fine-tuning or inference following the instructions at [SciTweets_SciBert @ Huggingface](https://huggingface.co/sschellhammer/SciTweets_SciBert).

## Publication:
<!-- TODO: Update with correct information once we uploaded the paper somewhere -->
Please cite the following paper if you are using the dataset, annotation framework or pretrained classifier

1. *Hafid, Salim, et al. "SciTweets-A Dataset and Annotation Framework for Detecting Scientific Online Discourse." Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022, [download](https://arxiv.org/abs/2206.07360).*

```bib
@inproceedings{hafid2022scitweets,
  title={SciTweets-A Dataset and Annotation Framework for Detecting Scientific Online Discourse},
  author={Hafid, Salim and Schellhammer, Sebastian and Bringay, Sandra and Todorov, Konstantin and Dietze, Stefan},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={3988--3992},
  year={2022}
}
```

## Licensing
This dataset is published under CC BY 4.0 license. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator (https://creativecommons.org/licenses/by/4.0/)

## Contact
Please contact salim.hafid@lirmm.fr or sebastian.schellhammer@gesis.org

## Acknowledgment
<!-- TODO: Update with french grant number -->
This work is supported by the AI4Sci grant, co-funded by MESRI (France) and BMBF (Germany, grant 01IS21086) and the French National Research Agency (ANR).
