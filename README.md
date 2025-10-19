# Machine Learning Interview Questions — Comprehensive Answers

## 1. How is machine learning different from general programming?

Machine learning differs fundamentally from traditional programming in how problems are solved. In general programming, developers write explicit rules and logic to transform inputs into outputs—a deterministic, rule-based approach. Machine learning, by contrast, learns patterns from data rather than following hardcoded instructions. ML systems can adapt and improve as they process more data, handling complex, unstructured inputs like images, text, or audio where explicit rules would be impractical to define. Traditional programming excels at well-defined, rule-based tasks with predictable outputs, while ML thrives in pattern recognition and prediction tasks where the relationship between inputs and outputs is too complex to program manually.[^1][^2][^3][^4][^5][^6][^7]

## 2. What is Overfitting in Machine Learning and how can it be avoided?

Overfitting occurs when a model learns noise and random fluctuations in training data rather than the underlying patterns, resulting in excellent training performance but poor generalization to new data. It can be prevented through several techniques: **early stopping** (halting training when validation performance degrades), **regularization** (L1/L2 penalties that constrain model complexity), **cross-validation** (testing on multiple data subsets), **pruning** (removing unnecessary features or model components), **data augmentation** (expanding training data artificially), and **ensemble methods** (combining multiple models). Using sufficient, diverse training data and selecting appropriate model complexity are also crucial. The key is balancing model complexity with the amount of available training data.[^8][^9][^10][^11][^12]

## 3. Why do we perform normalization?

Normalization scales features to a common range, typically \$\$ or standardizes them to have mean 0 and standard deviation 1. This prevents features with larger scales from dominating distance-based algorithms like k-nearest neighbors, support vector machines, or neural networks. Normalization accelerates convergence in gradient-based optimization by creating a more uniform loss surface, reducing training time significantly. It also prevents numerical instability issues like vanishing or exploding gradients in deep learning. Additionally, normalized data makes features more comparable and interpretable, ensuring that the learning algorithm treats all features with equal importance initially.[^13][^14][^15][^16][^17][^18]

## 4. What is the difference between precision and recall?

Precision measures the accuracy of positive predictions—the fraction of correctly identified positive instances among all instances predicted as positive. It answers: "Of all items labeled positive, how many are actually positive?" Precision = True Positives / (True Positives + False Positives). Recall (sensitivity) measures the model's ability to find all actual positive instances—the fraction of true positives among all actual positive instances. It answers: "Of all actual positive items, how many did we identify?" Recall = True Positives / (True Positives + False Negatives). There's an inherent tradeoff: increasing recall often decreases precision and vice versa. High precision is crucial when false positives are costly (e.g., spam filtering), while high recall is essential when missing positives is dangerous (e.g., disease diagnosis).[^19][^20][^21][^22]

## 5. What is the bias-variance tradeoff?

The bias-variance tradeoff is a fundamental concept describing the balance between two sources of prediction error. **Bias** refers to errors from oversimplified models that fail to capture underlying patterns, leading to underfitting. High bias models consistently make predictions far from actual values. **Variance** refers to errors from overly complex models that fit training data too closely, including noise, leading to overfitting. High variance models produce widely varying predictions across different training sets. As model complexity increases, bias decreases but variance increases. The goal is finding optimal complexity that minimizes total error by balancing both components. Techniques like regularization, cross-validation, and ensemble methods help achieve this balance.[^23][^24][^25][^26]

## 6. What is Principal Component Analysis?

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance. PCA identifies principal components—new orthogonal axes that capture the most important patterns in data. It works by computing eigenvectors and eigenvalues from the data's covariance matrix. The first principal component points in the direction of greatest variance, the second component (perpendicular to the first) captures the next highest variance, and so on. PCA standardizes data first, then projects it onto selected principal components, reducing dimensionality while retaining essential information. It's widely used for data visualization, noise reduction, feature extraction, and speeding up machine learning algorithms.[^27][^28][^29][^30]

## 7. What is one-shot learning?

One-shot learning is a machine learning paradigm where models learn to recognize objects or patterns from just a single training example (or very few examples). Unlike traditional deep learning requiring thousands of labeled samples, one-shot learning mimics human ability to recognize new concepts quickly from minimal exposure. It typically uses techniques like **metric learning** (learning similarity functions to compare new instances with known examples), **transfer learning** (building on pre-learned representations), or **siamese networks** (learning to assess similarities between pairs of inputs). One-shot learning is particularly valuable in computer vision for facial recognition, object detection, and scenarios where collecting large labeled datasets is impractical or expensive. Applications include signature verification, personalized recommendations, and medical imaging where rare conditions limit available training data.[^31][^32][^33][^34][^35][^36]

## 8. What is the difference between stochastic gradient descent (SGD) and gradient descent (GD)?

Gradient descent (GD), also called batch gradient descent, computes gradients using the entire training dataset before updating model parameters. Stochastic gradient descent (SGD) updates parameters after processing each individual training example (or small batch). **Key differences**: GD is slower but provides accurate, stable gradient estimates leading to smooth convergence. SGD is faster, requires less memory, and updates parameters more frequently, enabling quicker convergence. GD is deterministic (same results for same initial conditions), while SGD is stochastic due to random sample selection. SGD can escape local minima due to its noisy updates and handles large datasets more effectively. However, SGD has less accurate gradient estimates and more erratic convergence paths. In practice, mini-batch gradient descent (using small batches) balances both approaches.[^37][^38]

## 9. What is the Central Limit Theorem?

The Central Limit Theorem (CLT) states that as sample size increases, the distribution of sample means approaches a normal (bell-shaped) distribution, regardless of the underlying population distribution. For sufficiently large samples (typically n > 30), sample means will be normally distributed even if the original data is skewed or non-normal. The theorem specifies that the sample mean distribution has mean equal to the population mean (μ) and standard deviation equal to σ/√n, where σ is the population standard deviation and n is sample size. This standard deviation of sample means is called the standard error. CLT is foundational in statistics because it enables making inferences about populations using sample data, even when population distribution is unknown. It justifies using normal distribution-based methods in hypothesis testing and confidence interval construction.[^39][^40][^41][^42][^43]

## 10. Explain the working principle of SVM.

Support Vector Machines (SVM) are supervised learning algorithms that find the optimal hyperplane separating different classes by maximizing the margin between them. The **margin** is the distance from the hyperplane to the nearest data points (support vectors) from each class. SVM identifies the hyperplane that maximizes this margin, ensuring robust class separation. For linearly separable data, SVM finds a "hard margin" with no misclassifications. For non-linearly separable data, SVM uses a "soft margin" allowing some misclassifications while minimizing a penalty term (hinge loss). The objective function balances margin maximization with penalty minimization, controlled by hyperparameter λ. For non-linear problems, SVM applies the **kernel trick**, mapping data to higher dimensions where linear separation becomes possible. SVM is robust to outliers and effective in high-dimensional spaces.[^44][^45][^46][^47][^48]

## 11. What is the difference between L1 and L2 regularization? What is their significance?

L1 regularization (Lasso) adds the absolute value sum of coefficients as a penalty term to the loss function, while L2 regularization (Ridge) adds the squared sum of coefficients. **Key differences**: L1 can shrink coefficients to exactly zero, effectively performing feature selection and creating sparse models. L2 shrinks coefficients toward zero but rarely to exactly zero, maintaining all features with reduced influence. L1 is less effective with multicollinearity, potentially randomly selecting one correlated feature. L2 handles multicollinearity well by distributing coefficient values among correlated features. **Significance**: L1 improves model interpretability by eliminating irrelevant features, reducing complexity. L2 improves model stability and generalization without removing features entirely. Both prevent overfitting by penalizing model complexity, with the penalty strength controlled by hyperparameter λ.[^49][^50][^51][^52]

## 12. What is the purpose of splitting a given dataset into training and validation data?

Splitting data into training and validation sets is essential for building models that generalize well to unseen data. The **training set** is used to fit model parameters and learn patterns. The **validation set** is used during training to tune hyperparameters, select optimal model architectures, and detect overfitting before final evaluation. Using separate sets prevents data leakage—training on data you later evaluate on would produce artificially high accuracy metrics that don't reflect real-world performance. The validation set provides unbiased performance estimates during development, guiding decisions about model complexity, regularization strength, and when to stop training. A **test set** (third split) is reserved for final, unbiased evaluation after all development decisions are made. This three-way split ensures realistic assessment of model generalization capability.[^53][^54][^55][^56]

## 13. Why removing highly correlated features are considered a good practice?

Removing highly correlated features is beneficial because they carry redundant information, increasing model complexity without adding predictive value. **Multicollinearity** (high correlation between features) makes it difficult to determine individual feature effects on predictions, reducing model interpretability. It causes numerical instability in algorithms, particularly those relying on matrix inversions. Correlated features can lead to overfitting as models may capture noise in the small differences between highly similar features. Removing redundancy simplifies models, reduces training time, and decreases memory requirements. When features measure the same or similar underlying concepts, keeping only one preserves information while reducing dimensionality. This helps avoid the curse of dimensionality. Typically, when two features have correlation coefficients near ±1, one is removed based on domain knowledge, feature importance scores, or statistical properties.[^57][^58][^59][^60]

## 14. Reverse a linked list in place.

To reverse a linked list in place, use three pointers to iteratively reverse the direction of each node's next pointer. Initialize `previous = null`, `current = head`, and `next = null`. Iterate through the list: save the next node (`next = current.next`), reverse the current node's pointer (`current.next = previous`), advance previous and current pointers (`previous = current`, `current = next`). Continue until `current` becomes null. Finally, return `previous` as the new head. This algorithm has O(n) time complexity and O(1) space complexity since it only uses three pointers regardless of list size. The key insight is that you need to save the next node before modifying the current node's pointer, otherwise you lose access to the rest of the list. This is a classic algorithm problem testing pointer manipulation and understanding of linked list data structures.[^13]

## 15. What is the reason behind the curse of dimensionality?

The curse of dimensionality refers to various phenomena where algorithm efficiency and effectiveness deteriorate as data dimensionality increases exponentially. As dimensions increase, data points become increasingly sparse—the volume of high-dimensional space grows exponentially while the number of data points remains fixed. This sparsity makes it difficult to identify meaningful patterns or statistical relationships. The amount of data required for statistically sound predictions grows exponentially with dimensions. Distance metrics become less meaningful in high dimensions—all points appear equidistant, breaking distance-based algorithms like k-nearest neighbors. Computational complexity, training time, and memory requirements increase dramatically. The curse increases overfitting risk as models have more parameters to fit with relatively less data per dimension. Solutions include dimensionality reduction (PCA, feature selection), feature engineering, regularization, and collecting more data.[^61][^62][^63][^64][^65]

## 16. What is Linear Discriminant Analysis?

Linear Discriminant Analysis (LDA) is a supervised technique for classification and dimensionality reduction that finds linear combinations of features that best separate classes. LDA maximizes the ratio of between-class variance to within-class variance, creating decision boundaries that separate classes optimally. **Key assumptions**: data in each class follows a normal distribution, all classes have equal covariance matrices, and classes are linearly separable. LDA projects high-dimensional data onto a lower-dimensional space (typically one less than the number of classes) where class separation is maximized. It differs from PCA: PCA maximizes total variance without considering class labels, while LDA maximizes class separability. LDA is a generalization of Fisher's Linear Discriminant, extending it to handle multiple classes. Applications include face recognition, pattern recognition, medical diagnosis, and preprocessing for other machine learning algorithms.[^66][^67][^68][^69]

## 17. Can you explain the differences between supervised, unsupervised, and reinforcement learning?

**Supervised learning** trains on labeled data where input-output pairs are provided, learning to map inputs to known outputs. It solves classification (predicting categories) and regression (predicting continuous values) problems using algorithms like decision trees, SVM, and neural networks. Examples include medical diagnosis, fraud detection, and sentiment analysis. **Unsupervised learning** discovers patterns in unlabeled data without predefined outputs. It performs clustering (grouping similar data) and association (finding relationships) using algorithms like K-means, hierarchical clustering, and PCA. Applications include customer segmentation, anomaly detection, and dimensionality reduction. **Reinforcement learning** trains agents to make sequential decisions by interacting with environments, learning from rewards and penalties rather than labeled examples. It uses trial-and-error with algorithms like Q-learning and Deep Q-Networks to optimize long-term rewards. Applications include robotics, autonomous driving, and game playing.[^70][^71][^72][^73][^74]

## 18. What are convolutional networks? Where can we use them?

Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed primarily for processing grid-structured data like images. CNNs use convolutional layers that apply learnable filters (kernels) sliding across inputs to automatically extract hierarchical features, mimicking the human visual cortex organization. Unlike fully connected networks, CNNs exploit spatial structure through **local connectivity** (neurons connect only to small regions), **parameter sharing** (same filter used across entire input), and **translation equivariance** (detecting patterns regardless of position). **Applications** include: image classification, object detection and localization (autonomous vehicles, surveillance), semantic segmentation (medical imaging, scene understanding), facial recognition (security systems), medical diagnosis (tumor detection, disease classification), image generation (GANs, style transfer), video analysis, natural language processing, recommendation systems, and financial time series analysis. CNNs require less preprocessing than traditional algorithms as they learn optimal filters automatically.[^75][^76][^77][^78][^79]

## 19. What is cost function?

A cost function (loss function) is a mathematical formula that quantifies how well a machine learning model performs by measuring the difference between predicted and actual values. It calculates the total error across all predictions, providing a numerical representation of model accuracy. The primary objective during training is to **minimize** the cost function by adjusting model parameters through optimization algorithms like gradient descent. Different cost functions suit different tasks: **Mean Squared Error (MSE)** for regression (calculating average squared differences between predictions and actuals), **Cross-Entropy** for classification (measuring dissimilarity between predicted probabilities and true labels), and **Hinge Loss** for support vector machines. The cost function guides the learning process—its gradient indicates the direction to adjust parameters for improved performance. Lower cost values indicate better model fit, though extremely low training cost may signal overfitting.[^80][^81][^82][^83][^84]

## 20. List different activation neurons or functions.

Common activation functions include: **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$, most popular for hidden layers, computationally efficient, helps mitigate vanishing gradients but suffers from dying ReLU problem. **Sigmoid**: $f(x) = 1/(1 + e^{-x})$, outputs between 0 and 1, used in binary classification output layers, suffers from vanishing gradient problem. **Tanh (Hyperbolic Tangent)**: $f(x) = (e^x - e^{-x})/(e^x + e^{-x})$, outputs between -1 and 1, zero-centered making optimization easier than sigmoid. **Leaky ReLU**: $f(x) = \max(\alpha x, x)$ where α is small (e.g., 0.01), addresses dying ReLU problem by allowing small negative values. **Softmax**: $f(x_i) = e^{x_i}/\sum e^{x_j}$, converts outputs to probability distribution, used in multi-class classification output layers. **ELU (Exponential Linear Unit)**: smooth approximation to ReLU with negative values. **Swish**: $f(x) = x \cdot \text{sigmoid}(x)$, discovered by Google, outperforms ReLU in deep networks. Each function has specific use cases based on problem type and layer position.[^76][^75]

## 21. Explain Epoch vs. Batch vs. Iteration.

An **epoch** is one complete pass through the entire training dataset—when every training example has been used once to update the model. An **iteration** is a single update of model parameters, typically using one batch of data. A **batch** is a subset of training examples processed together before updating parameters. The relationship: If you have 1,000 training examples and a batch size of 100, one epoch contains 10 iterations (1,000 ÷ 100 = 10 batches). **Batch size** determines the number of samples processed before parameter updates—larger batches provide more accurate gradient estimates but require more memory and computation. Small batches (or single samples in SGD) enable faster, more frequent updates but with noisier gradients. Training typically requires multiple epochs, with the model gradually improving as it sees the data repeatedly. The number of epochs needed depends on dataset size, model complexity, and learning rate.[^38][^37]

## 22. What is regularization, why do we use it, and give some examples of common methods?

Regularization is a set of techniques that prevent overfitting by adding constraints or penalties to discourage model complexity. It helps models generalize better to unseen data by reducing their capacity to memorize training data noise. **Why use it**: Complex models with many parameters easily overfit training data, performing poorly on new data. Regularization balances model complexity with training accuracy, creating simpler, more robust models. **Common methods**: **L1 regularization (Lasso)** adds absolute value of coefficient magnitudes to the loss function, performing feature selection by shrinking some coefficients to zero. **L2 regularization (Ridge)** adds squared coefficient magnitudes, shrinking all coefficients without eliminating features. **Dropout** randomly deactivates neurons during training, preventing co-adaptation. **Early stopping** halts training when validation performance stops improving. **Data augmentation** artificially expands training data. **Elastic Net** combines L1 and L2 penalties.[^51][^52][^49][^76][^8]

## 23. Explain why the performance of XGBoost is better than that of SVM?

XGBoost (Extreme Gradient Boosting) often outperforms SVM for several reasons. **Ensemble approach**: XGBoost builds multiple weak learners (decision trees) sequentially, each correcting previous errors, while SVM finds a single optimal hyperplane. This ensemble method captures complex non-linear patterns more effectively. **Handling heterogeneous data**: XGBoost naturally handles mixed feature types (categorical, numerical) and missing values, whereas SVM requires extensive preprocessing and complete data. **Scalability**: XGBoost is optimized for speed with parallel processing and efficient memory usage, handling large datasets better than SVM which has O(n²) to O(n³) complexity. **Feature interactions**: XGBoost automatically captures feature interactions through tree splits, while SVM requires manual kernel selection. **Regularization**: XGBoost includes built-in L1/L2 regularization controlling tree complexity, preventing overfitting. **Interpretability**: XGBoost provides feature importance scores aiding model understanding. **Tuning**: XGBoost offers more hyperparameters for fine-tuning performance. However, SVM excels in high-dimensional spaces with clear margins and smaller datasets.[^45][^48][^70]

## 24. What is the difference between correlation and causality?

**Correlation** measures statistical association between variables—when one variable changes, the other tends to change in a predictable pattern. Correlation coefficients range from -1 to +1, indicating strength and direction of relationships. **Causality** means one variable directly causes changes in another through a causal mechanism. **Key difference**: Correlation does NOT imply causation—two variables can be correlated without one causing the other. Common scenarios creating non-causal correlation: **confounding variables** (third variable influences both, like ice cream sales and drowning rates both increase in summer), **reverse causation** (effect mistaken for cause), **coincidence** (spurious correlation in random data). Establishing causality requires: temporal precedence (cause precedes effect), elimination of alternative explanations, controlled experiments or strong observational evidence, and theoretical mechanism explaining the relationship. Randomized controlled trials are the gold standard for proving causation by controlling confounding factors. In machine learning, predictive models can exploit correlations without understanding causal relationships, but causal inference requires specialized techniques.[^20][^21][^19]

## 25. What is stemming?

Stemming is a text preprocessing technique in natural language processing that reduces words to their root or base form (stem) by removing suffixes and prefixes. For example, "running," "runs," and "ran" all stem to "run." Stemming uses algorithmic rules to chop word endings, creating stems that may not be valid dictionary words. The most common algorithm is the **Porter Stemmer**, which applies a series of rule-based transformations. Other algorithms include Snowball Stemmer and Lancaster Stemmer, each with different aggressiveness levels. **Purpose**: Stemming reduces vocabulary size, treating morphological variants of words as the same token, improving information retrieval and text analysis efficiency. It helps match related words in search queries and document indexing. **Limitations**: Stemming can be crude, producing non-words or incorrectly grouping unrelated words. For example, "university" and "universe" might stem to the same root despite different meanings. Stemming is faster but less accurate than lemmatization, making it suitable for applications prioritizing speed over linguistic accuracy.[^13]

## 26. What is Lemmatization?

Lemmatization is a sophisticated text preprocessing technique that reduces words to their dictionary base form (lemma) using vocabulary and morphological analysis. Unlike stemming, lemmatization produces valid words by considering context and part-of-speech. For example, "better" lemmatizes to "good" (adjective), "running" to "run" (verb), and "am/is/are" all to "be." Lemmatization uses linguistic knowledge and dictionaries, often employing tools like WordNet. **Differences from stemming**: Lemmatization produces linguistically correct base forms while stemming uses crude rule-based chopping. Lemmatization considers word context and part-of-speech; stemming doesn't. Lemmatization is slower but more accurate; stemming is faster but less precise. **Example**: "studies" → lemmatization produces "study," stemming might produce "studi." **Applications**: Lemmatization is preferred when linguistic accuracy matters—text classification, sentiment analysis, semantic search, and chatbots. It's essential when word meaning must be preserved. However, for applications like information retrieval where speed matters more than precision, stemming suffices.[^13]

## 27. What is Static Memory Allocation?

Static memory allocation is a memory management technique where memory is allocated at compile time before program execution begins. The size, location, and lifetime of statically allocated variables are determined during compilation and remain fixed throughout program execution. **Characteristics**: Memory is allocated from the stack or static data segment, variables have fixed sizes known at compile time, memory is automatically deallocated when variables go out of scope, and allocation/deallocation is faster than dynamic allocation. **Examples**: Global variables, static variables, and fixed-size arrays are statically allocated. **Advantages**: Fast access (no runtime allocation overhead), predictable memory usage, no fragmentation issues, and automatic cleanup. **Disadvantages**: Inflexible size (cannot change at runtime), potential memory waste if overallocated or stack overflow if underestimated, and limited to compile-time known sizes. **Contrast with dynamic allocation**: Dynamic allocation uses heap memory, occurs at runtime with functions like malloc/new, allows flexible sizes based on runtime conditions, but requires manual deallocation and is slower.[^13]

## 28. What are some tools used to discover outliers?

**Statistical methods**: **Z-score** (standardized distance from mean; values >3 or <-3 standard deviations are outliers), **Interquartile Range (IQR)** (values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR), **Modified Z-score** using median absolute deviation for robustness. **Visualization techniques**: **Box plots** show quartiles and mark outliers beyond whiskers, **scatter plots** reveal unusual data points, **histograms** show distribution tails. **Distance-based methods**: **K-Nearest Neighbors (KNN)** identifies points far from neighbors, **DBSCAN** clustering marks points not belonging to dense clusters as outliers. **Machine learning approaches**: **Isolation Forest** isolates anomalies using random partitioning, **One-Class SVM** learns normal data boundaries, **Local Outlier Factor (LOF)** compares local density with neighbors, **Autoencoders** identify points with high reconstruction error. **Domain-specific methods**: **Grubbs' test** for normally distributed data, **Cook's distance** for regression influence. **Tools/libraries**: Python's scikit-learn, PyOD, pandas; R's outliers package; visualization tools like Tableau.[^58][^62][^13]

## 29. What are some methods to improve inference time?

**Model optimization**: **Quantization** reduces precision (32-bit to 8-bit integers), decreasing model size and computation. **Pruning** removes unnecessary weights/neurons with minimal accuracy loss. **Knowledge distillation** trains smaller "student" models to mimic larger "teacher" models. **Model compression** combines techniques to reduce size while maintaining performance. **Architecture choices**: Use efficient architectures like MobileNet, EfficientNet, or SqueezeNet designed for speed. Replace complex operations with faster alternatives. **Hardware acceleration**: Use GPUs, TPUs, or specialized AI chips for parallel processing. **Batch processing**: Process multiple inputs simultaneously rather than individually. **Caching**: Store and reuse intermediate computations for repeated inputs. **Early stopping**: Terminate computation when confidence threshold is reached. **Approximate methods**: Use approximate nearest neighbor search instead of exact search. **Preprocessing**: Optimize data pipelines, reduce input resolution where acceptable. **Framework optimization**: Use optimized inference engines like TensorRT, ONNX Runtime, or TensorFlow Lite. **Parallelization**: Distribute computation across multiple cores/devices.[^79][^75][^76][^13]
<span style="display:none">[^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^110][^111][^112][^113][^114][^115][^116][^117][^118][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://insightsoftware.com/blog/machine-learning-vs-traditional-programming/

[^2]: https://www.linkedin.com/pulse/how-machine-learning-different-from-general-programming-zlcuf

[^3]: https://www.geeksforgeeks.org/machine-learning/traditional-programming-vs-machine-learning/

[^4]: https://unp.education/content/how-machine-learning-is-different-from-general-programming/

[^5]: https://sparks.codezela.com/machine-learning-vs-programming/

[^6]: https://www.avenga.com/magazine/machine-learning-programming/

[^7]: https://www.institutedata.com/blog/machine-learning-vs-traditional-programming-choosing-the-right-approach-for-your-projects/

[^8]: https://aws.amazon.com/what-is/overfitting/

[^9]: https://elitedatascience.com/overfitting-in-machine-learning

[^10]: https://developers.google.com/machine-learning/crash-course/overfitting/overfitting

[^11]: https://www.ibm.com/think/topics/overfitting

[^12]: https://www.geeksforgeeks.org/machine-learning/underfitting-and-overfitting-in-machine-learning/

[^13]: https://github.com/loukesio/Machine-Learning

[^14]: https://www.geeksforgeeks.org/machine-learning/what-is-data-normalization/

[^15]: https://www.datacamp.com/tutorial/normalization-in-machine-learning

[^16]: https://developers.google.com/machine-learning/crash-course/numerical-data/normalization

[^17]: https://www.deepchecks.com/glossary/normalization-in-machine-learning/

[^18]: https://en.wikipedia.org/wiki/Normalization_(machine_learning)

[^19]: https://www.deepchecks.com/precision-vs-recall-in-the-quest-for-model-mastery/

[^20]: https://en.wikipedia.org/wiki/Precision_and_recall

[^21]: https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall

[^22]: https://builtin.com/data-science/precision-and-recall

[^23]: https://h2o.ai/wiki/bias-variance-tradeoff/

[^24]: https://uniathena.com/understanding-bias-variance-tradeoff-balance-model-performance

[^25]: https://serokell.io/blog/bias-variance-tradeoff

[^26]: https://www.geeksforgeeks.org/machine-learning/ml-bias-variance-trade-off/

[^27]: https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/

[^28]: https://builtin.com/data-science/step-step-explanation-principal-component-analysis

[^29]: https://en.wikipedia.org/wiki/Principal_component_analysis

[^30]: https://www.ibm.com/think/topics/principal-component-analysis

[^31]: https://www.geeksforgeeks.org/machine-learning/one-shot-learning-in-machine-learning-1/

[^32]: https://encord.com/blog/one-shot-learning-guide/

[^33]: https://toloka.ai/blog/teaching-machines-with-minimal-data-one-shot-learning/

[^34]: https://www.eimt.edu.eu/what-is-one-shot-learning-approach-in-machine-learning

[^35]: https://serokell.io/blog/nn-and-one-shot-learning

[^36]: https://www.dremio.com/wiki/one-shot-learning/

[^37]: https://www.geeksforgeeks.org/machine-learning/difference-between-batch-gradient-descent-and-stochastic-gradient-descent/

[^38]: https://www.geeksforgeeks.org/machine-learning/ml-stochastic-gradient-descent-sgd/

[^39]: https://www.geeksforgeeks.org/maths/central-limit-theorem/

[^40]: https://www.datacamp.com/tutorial/central-limit-theorem

[^41]: https://en.wikipedia.org/wiki/Central_limit_theorem

[^42]: https://www.scribbr.com/statistics/central-limit-theorem/

[^43]: https://statisticsbyjim.com/basics/central-limit-theorem/

[^44]: https://www.coursera.org/articles/svm

[^45]: https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/

[^46]: https://en.wikipedia.org/wiki/Support_vector_machine

[^47]: https://www.mathworks.com/discovery/support-vector-machine.html

[^48]: https://www.ibm.com/think/topics/support-vector-machine

[^49]: https://builtin.com/data-science/l2-regularization

[^50]: https://wandb.ai/mostafaibrahim17/ml-articles/reports/Understanding-L1-and-L2-regularization-techniques-for-optimized-model-training--Vmlldzo3NzYwNTM5

[^51]: https://www.tutorialspoint.com/difference-between-l1-and-l2-regularization

[^52]: https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization

[^53]: https://encord.com/blog/train-val-test-split/

[^54]: https://builtin.com/data-science/train-test-split

[^55]: https://mlu-explain.github.io/train-test-validation/

[^56]: https://www.v7labs.com/blog/train-validation-test-set

[^57]: https://www.reddit.com/r/learnmachinelearning/comments/s60y34/how_to_remove_correlating_features/

[^58]: https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python

[^59]: https://campus.datacamp.com/courses/dimensionality-reduction-in-python/feature-selection-i-selecting-for-feature-information?ex=13

[^60]: https://www.rohan-paul.com/p/ml-interview-q-series-when-would-ff2

[^61]: https://zilliz.com/glossary/curse-of-dimensionality-in-machine-learning

[^62]: https://www.geeksforgeeks.org/machine-learning/curse-of-dimensionality-in-machine-learning/

[^63]: https://en.wikipedia.org/wiki/Curse_of_dimensionality

[^64]: https://www.datacamp.com/blog/curse-of-dimensionality-machine-learning

[^65]: https://towardsdatascience.com/curse-of-dimensionality-an-intuitive-exploration-1fbf155e1411/

[^66]: https://www.r-bloggers.com/2024/02/understanding-linear-discriminant-analysis-lda/

[^67]: https://www.geeksforgeeks.org/machine-learning/ml-linear-discriminant-analysis/

[^68]: https://en.wikipedia.org/wiki/Linear_discriminant_analysis

[^69]: https://web.stanford.edu/class/stats202/notes/Classification/LDA.html

[^70]: https://www.geeksforgeeks.org/machine-learning/supervised-vs-reinforcement-vs-unsupervised/

[^71]: https://www.aitude.com/supervised-vs-unsupervised-vs-reinforcement/

[^72]: https://www.educative.io/answers/supervised-vs-unsupervised-vs-reinforcement-learning

[^73]: https://www.phdata.io/blog/difference-between-supervised-unsupervised-reinforcement-learning/

[^74]: https://www.ibm.com/think/topics/supervised-vs-unsupervised-learning

[^75]: https://encord.com/blog/convolutional-neural-networks-explained/

[^76]: https://en.wikipedia.org/wiki/Convolutional_neural_network

[^77]: https://www.xenonstack.com/blog/convolutional-neural-network

[^78]: https://www.flatworldsolutions.com/data-science/articles/7-applications-of-convolutional-neural-networks.php

[^79]: https://www.ibm.com/think/topics/convolutional-neural-networks

[^80]: https://www.alooba.com/skills/concepts/machine-learning/cost-functions/

[^81]: https://www.appliedaicourse.com/blog/cost-function-in-machine-learning/

[^82]: https://builtin.com/machine-learning/cost-function

[^83]: https://www.simplilearn.com/tutorials/machine-learning-tutorial/cost-function-in-machine-learning

[^84]: https://www.geeksforgeeks.org/machine-learning/ml-cost-function-in-logistic-regression/

[^85]: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

[^86]: https://www.coursera.org/articles/precision-vs-recall-machine-learning

[^87]: https://encord.com/blog/classification-metrics-accuracy-precision-recall/

[^88]: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

[^89]: https://en.wikipedia.org/wiki/Bias–variance_tradeoff

[^90]: https://www.v7labs.com/blog/precision-vs-recall-guide

[^91]: https://www.cs.cmu.edu/~elaw/papers/pca.pdf

[^92]: https://www.geeksforgeeks.org/machine-learning/precision-and-recall-in-machine-learning/

[^93]: https://sebastianraschka.com/faq/docs/few-shot.html

[^94]: https://en.wikipedia.org/wiki/One-shot_learning_(computer_vision)

[^95]: https://aleksandarhaber.com/easy-to-understand-explanation-of-the-stochastic-gradient-descent-algorithm-with-python-implementation-from-scratch/

[^96]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

[^97]: https://arxiv.org/abs/2201.08815

[^98]: https://sebastianraschka.com/faq/docs/gradient-optimization.html

[^99]: https://www.miquido.com/ai-glossary/one-shot-learning/

[^100]: https://scikit-learn.org/stable/modules/svm.html

[^101]: https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf

[^102]: https://course.ccs.neu.edu/cs5100f11/resources/jakkula.pdf

[^103]: https://www.reddit.com/r/MachineLearning/comments/dgog2h/d_why_is_l2_preferred_over_l1_regularization/

[^104]: https://www.youtube.com/watch?v=_YPScrckx28

[^105]: https://global.trocco.io/blogs/why-do-you-split-data-into-training-and-validation-sets

[^106]: https://www.techtarget.com/whatis/definition/support-vector-machine-SVM

[^107]: https://www.reddit.com/r/datascience/comments/hx04uf/how_to_remove_correlated_features/

[^108]: https://stackoverflow.com/questions/18270899/remove-highly-correlated-components

[^109]: https://stackoverflow.com/questions/75380186/how-to-drop-one-of-any-two-highly-correlated-features-having-low-correlation-wit

[^110]: https://www.diva-portal.org/smash/get/diva2:1632660/FULLTEXT01.pdf

[^111]: https://www.youtube.com/watch?v=FndwYNcVe0U

[^112]: https://www.youtube.com/watch?v=azXCzI57Yfc

[^113]: https://www.youtube.com/watch?v=ZGbXSVZjES4

[^114]: https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/

[^115]: https://www.youtube.com/watch?v=1FZ0A1QCMWc

[^116]: https://www.pecan.ai/blog/3-types-of-machine-learning/

[^117]: https://www.vervecopilot.com/question-bank/difference-between-supervised-unsupervised-reinforcement-learning

[^118]: https://www.linkedin.com/pulse/demystifying-machine-learning-supervised-unsupervised-bushra-akram

