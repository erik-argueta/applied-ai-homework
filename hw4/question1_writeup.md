# Homework 4 · Question 1 — AG News Topic Classification

## 1(f) Documents, corpus, and feature matrices

A **document** is on article in the AG News dataset represented by its title and description. The **corpus** is the collection of documents. AG News has 120,000 training documents and 7,600 test documents across World, Sports, Business, and Sci/Tech.

In a **document-term matrix (DTM)**, each row represnts one document, and column represents one term in the vocabulary learned from the training corpus. A cell in the count DTM records how many times that term occurs in that document. The coressponding TF-IDF cell contains a weight base on the term's frequency in the document and its rarity across the training corpus. The default TF-IDF vectorizer also applies L2 normalization to each document vector.

Each article contains only a small fraction of the vocabulary, so most cells are zero in both matrices. The training count DTM has 120,000 rows and 65,006 columns, with 3,739,321 nonzero entries out of 7,800,720,000 cells. That is **99.9521% sparsity**. Sparse storage saves memory by keeping the nonzero values and their positions, and sparse matrix operations can skip work on zero entries.

## 2. Train and evaluate an MLP classifier

The baseline MLP used `CountVectorizer` features with stop-word removal disabled. It used the provided AG News split, with 30,000 training documents and 1,900 test documents per category. The vocabulary was fitted on the training text, and the test text was transformed using that vocabulary.

| Setting               | Value                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Architecture          | 65,006 input features → one hidden layer of 32 units → 4 output units                                                                      |
| Activations           | ReLU in the hidden layer and softmax at the output                                                                                         |
| Optimizer             | Adam                                                                                                                                       |
| Loss                  | Multiclass cross-entropy (log loss), with default L2 regularization (`alpha=0.0001`)                                                       |
| Initial learning rate | `0.001`                                                                                                                                    |
| Batch size            | 1,024                                                                                                                                      |
| Epochs                | 10 completed (`max_iter=10`)                                                                                                               |
| Other settings        | `random_state=42`, `n_iter_no_change=10`, sparse features cast to `float32`, and the remaining MLP settings at their scikit-learn defaults |

The run reached its 10-epoch limit and reported a convergence warning. The metrics below are from the model at the end of epoch 10. The terminal heading prints `min_df=2`, but the current code constructs the baseline with `CountVectorizer()`, whose default is `min_df=1`. The comparison vectorizers inherit that setting from the baseline.

The held-out test accuracy was **91.18%** (6,930 of 7,600 correct). **Macro-averaged precision was 91.17%**. Precision was computed separately for each class as the fraction of predictions for that class that were correct, then averaged with equal weight across the four classes (`average="macro"`, `zero_division=0`).

### Confusion matrix

Rows are true categories and columns are predicted categories. This is the count-based model with stop-word removal disabled.

| True ↓ / Predicted → | World | Sports | Business | Sci/Tech |
| -------------------- | ----: | -----: | -------: | -------: |
| World                | 1,719 |     49 |       84 |       48 |
| Sports               |    30 |  1,851 |       13 |        6 |
| Business             |    64 |     13 |    1,663 |      160 |
| Sci/Tech             |    57 |     11 |      135 |    1,697 |

### Three misclassified test examples

| Article                                                                                                             | True category | Predicted category | Plausible explanation                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| _Super ant colony hits Australia_ — A giant ant colony discovered in Melbourne could threaten local insect species. | Sci/Tech      | World              | The references to Australia and Melbourne may make the story resemble international news even though it concerns ecology. |
| _IBM to hire even more new workers_ — IBM expected to reach its highest headcount since 1991.                       | Sci/Tech      | Business           | Hiring and headcount are common business topics, so the story can read as a workforce update about a technology company.  |
| _Some People Not Eligible to Get in on Google IPO_ — The story discusses who could participate in Google's IPO.     | Sci/Tech      | Business           | Terms such as “IPO,” “Wall Street,” “shares,” and “underwriters” give the article strong financial cues.                  |

### Why documents are the observations

Each training label belongs to an article, so each document is one supervised observation and the terms are its input features. Transposing the DTM makes each term a row and each document a column. A classifier fitted to those rows would need one label per term and would predict a category for a term based on its occurrences across documents. The original labels are defined per document, and a term can occur in articles from several categories.

## 3. Compare feature representations and stop-word removal

All four runs used the same AG News train/test split, one 32-unit hidden layer, four output classes, and the training settings listed in Part 2. The input width followed the vocabulary size for each run. Each vectorizer was fitted on the training text only, then used to transform the held-out test text. The stop-word removal runs used scikit-learn's built-in English stop-word list.

| Vectorizer        | Stop-word removal | Test accuracy | Test macro-precision |
| ----------------- | ----------------- | ------------: | -------------------: |
| `CountVectorizer` | No                |        91.18% |               91.17% |
| `TfidfVectorizer` | No                |    **91.93%** |           **91.93%** |
| `CountVectorizer` | Yes               |        91.05% |               91.04% |
| `TfidfVectorizer` | Yes               |        91.78% |               91.77% |

The confusion matrix in Part 2 corresponds to the first row of this comparison.

TF-IDF improved accuracy by 0.75 percentage points with stop words retained and 0.73 points with stop words removed, using the displayed scores. Macro-precision improved by similar amounts. One plausible explanation is that TF-IDF downweights terms that occur in many articles and gives more relative weight to terms that distinguish topics.

Removing stop words lowered accuracy and macro-precision slightly for both representations. TF-IDF already reduces the influence of many common words, so removing them may provide little extra benefit. Some of those words may also carry weak category signals. These differences describe this split and training run.
