# ED318 Statistical Machine Learning Lab Exam
**Roll:** \<roll_number\> &nbsp;&nbsp; **Name:** \<name\> &nbsp;&nbsp; **Set:** 1

---

## 1. Problem Statement

**Q1 — Generative vs Discriminative (Wine Dataset):**
A scientist claims all classification problems are fundamentally generative. We test this by comparing Gaussian generative models (LDA, QDA) against a discriminative model (Logistic Regression with SGD) on the Wine dataset (UCI), which has 178 samples, 13 chemical features, and 3 classes.

**Q2 — Reverse Engineering Oracle (Breast Cancer Dataset):**
A hospital AI system predicts tumor malignancy. We identify which learning principle — risk minimization or generalization — best explains its behavior by comparing five classifiers on the Breast Cancer Wisconsin dataset (UCI), which has 569 samples, 30 features, and 2 classes (212 malignant, 357 benign).

---

## 2. Methodology

### 2.1 Preprocessing

- **Train/test split:** 80/20 stratified random split (seed fixed for reproducibility).
- **Standardization:** Features zero-centered and unit-variance normalized using training statistics only (applied to both splits to avoid data leakage).
- **PCA (Q2 only):** Covariance matrix computed on training data; top 10 principal components retained (capturing ~95% variance), and the same projection applied to the test set.

### 2.2 Learning Algorithms

**Q1:**

| Model | Type | Key detail |
|---|---|---|
| LDA | Generative | Shared within-class covariance; Bayes decision boundary |
| QDA | Generative | Per-class covariance; quadratic boundary |
| Logistic Regression | Discriminative | One-vs-rest; L2 regularization; SGD optimizer |

**Q2:**

| Model | Principle | Key detail |
|---|---|---|
| Gaussian Naive Bayes | Generative | Feature independence assumption; class-conditional Gaussians |
| LDA | Generative | Linear decision boundary; shared covariance |
| Logistic Regression | Discriminative | L2 regularization; binary SGD |
| Linear SVM | ERM / margin | Hinge loss; subgradient SGD; regularization via weight decay |
| RBF SVM | Kernel / generalization | Random Fourier features (D=300) approximate RBF kernel; trained as linear SVM on feature map |

All models implemented from scratch using NumPy only (no scikit-learn).

### 2.3 Evaluation Strategy

- **Accuracy:** Fraction of correctly classified test samples.
- **ROC-AUC (Q2):** Area under the ROC curve computed using the Wilcoxon-Mann-Whitney statistic — equals the probability that a randomly chosen positive sample scores higher than a randomly chosen negative.
- Models compared with and without PCA to assess the effect of dimensionality reduction.

---

## 3. Experimental Results

### Q1 — Wine Dataset

| Model | Test Accuracy |
|---|---|
| LDA (Generative) | 1.0000 |
| QDA (Generative) | 1.0000 |
| Logistic Regression (Discriminative) | 1.0000 |

Both generative and discriminative models achieve perfect accuracy on Wine, suggesting the classes are linearly well-separated in the original feature space.

### Q2 — Breast Cancer Dataset

**Full features (30 dimensions):**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Naive Bayes | 0.9027 | 0.9787 |
| LDA | 0.9381 | 0.9936 |
| Logistic Regression | 0.9912 | 0.9996 |
| Linear SVM | 0.9292 | 0.9879 |
| RBF SVM | 0.6726 | 0.8986 |

**After PCA (10 components):**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Naive Bayes | 0.8850 | 0.9474 |
| LDA | 0.9381 | 0.9936 |
| Logistic Regression | 0.9912 | 0.9986 |
| Linear SVM | 0.9292 | 0.9879 |
| RBF SVM | 0.6726 | 0.9221 |

---

## 4. Discussion

**Q1:** Both generative (LDA, QDA) and discriminative (Logistic Regression) models reach 100% accuracy on Wine. This does not confirm that classification problems are "fundamentally generative." When the class distributions are approximately Gaussian, generative models work well — but they rely on distributional assumptions that may not hold in general. Discriminative models make fewer assumptions and directly minimize classification error, which often makes them more robust on real-world data where the true distribution is unknown.

**Q2:** Logistic Regression achieves the best accuracy (0.9912) and near-perfect AUC (0.9996), suggesting the Oracle is most likely a regularized discriminative model that balances empirical risk minimization with generalization through L2 regularization. LDA performs consistently (0.9381) with and without PCA, confirming that the first 10 principal components carry most of the discriminative signal. Naive Bayes is the weakest due to its feature independence assumption, which is violated here (many breast cancer features are correlated). The RBF SVM underperforms, likely because the random Fourier feature approximation required more features (larger D) or better hyperparameter tuning. PCA has minimal impact on most models, confirming the data is effectively low-dimensional despite having 30 features.

---
