# Video Presentation Script

Target duration: **18-20 minutes**. Maximum permitted duration: 25 minutes.

Before recording, execute the notebook on the real ULB dataset and replace every
`[RUN OUTPUT: ...]` marker with the generated result.

## Slide 1 - Project and objective

Hello, I am Bijitesh Kanrar. My end-course project is Anomaly Detection in
Financial Transactions, under the Banking and Finance category.

The objective is to identify fraudulent card transactions in a highly imbalanced
dataset while controlling the operational cost of false alerts. I compare four
anomaly-detection approaches and a supervised benchmark. I then select a
business-aware threshold, evaluate it on an untouched time-based test set, and
package the selected detector for deployment through a Flask API.

The central idea is that fraud detection is not only a model-accuracy problem.
It is a risk-control decision involving fraud capture, investigation capacity,
customer inconvenience, and governance.

## Slide 2 - Why fraud detection is difficult

Fraud is rare. In this dataset, only 492 of 284,807 transactions are fraudulent,
which is approximately 0.17 percent.

This creates a major evaluation trap. A model that labels every transaction as
normal would achieve more than 99 percent accuracy but would detect no fraud.
Therefore, I do not use accuracy for model selection. I focus on precision,
recall, PR-AUC, F2 score, false-positive rate, and the confusion-matrix counts.

There is also an operational trade-off. Low thresholds catch more fraud but send
more genuine transactions for investigation. High thresholds reduce workload but
can miss costly fraud. My methodology makes that trade-off explicit.

## Slide 3 - Dataset

I use the anonymized credit-card fraud dataset created through collaboration
between Worldline and the Machine Learning Group of Universite Libre de
Bruxelles and distributed through Kaggle.

It records European card transactions over two days. The fields V1 through V28
are PCA-anonymized attributes. Time is the number of elapsed seconds, Amount is
the transaction value, and Class is the label, where one means fraud.

The data protects commercial and customer details, but anonymization limits
business explanation. Also, a two-day historical sample is suitable for an
academic benchmark, not immediate production deployment.

## Slide 4 - End-to-end workflow

My workflow has eight stages.

First, I validate the schema and audit missing values and duplicates. Second, I
clean the data without deleting unusual transactions merely because they look
extreme. Third, I engineer a log amount and cyclical time-of-day features.

Fourth, I create time-ordered train, validation, and test partitions. Fifth, I fit
preprocessing and candidate models on training data only. Sixth, I select each
model's decision threshold on validation data using an explicit cost function.
Seventh, I evaluate the locked thresholds once on the test set. Finally, I save
the selected model, scaler, feature order, and threshold in one deployment
bundle and expose it through an API.

## Slide 5 - Preprocessing and feature engineering

The pipeline first checks that Time, V1 to V28, Amount, and Class are available
and numeric. It records missing values and exact duplicates before removing rows
that cannot be used.

I retain unusual valid transactions because those are precisely what the project
needs to detect. Removing outliers before anomaly modelling would destroy useful
fraud signals.

Transaction amount is highly right-skewed, so I apply log one plus amount. I
also convert time-of-day into sine and cosine features. This treats 11:59 PM and
12:01 AM as close rather than far apart. Finally, I use RobustScaler, fitted only
on training data, to reduce scale dominance without allowing validation or test
information into model fitting.

## Slide 6 - Leakage-resistant experiment

Instead of a purely random split, I sort transactions by time and create 70
percent training, 15 percent validation, and 15 percent test partitions.

This better represents the real use case: a model learns from earlier activity
and scores later transactions. The training partition fits transformations and
models. The validation partition compares models and selects thresholds. The
test partition remains untouched until all decisions are locked.

The anomaly models learn from the training records labelled normal. Technically,
this is novelty detection rather than completely label-free unsupervised
learning. I state that distinction explicitly because labels define the clean
reference population, although fraud labels are not part of the detector's loss
function.

## Slide 7 - Candidate anomaly models

I compare four complementary methods.

The first is an IQR amount baseline. It is transparent and checks how far a
log-amount is from the training median, measured in interquartile ranges. Its
weakness is that it is univariate.

The second is Isolation Forest. It repeatedly creates random partitions. Rare
observations are isolated quickly and receive higher anomaly scores.

The third is Local Outlier Factor in novelty mode. It compares an observation's
local density with its neighbours and can identify locally unusual behaviour.

The fourth is PCA reconstruction error. PCA learns a lower-dimensional subspace
from normal transactions. Transactions that reconstruct poorly are treated as
more anomalous.

These models represent statistical, global isolation, local density, and
reconstruction-based approaches.

I fine-tune a compact grid within each model family. Isolation Forest varies the
number of trees and sample fraction, Local Outlier Factor varies neighbourhood
size, and PCA varies retained variance. Selection uses validation cost, with F2
and PR-AUC as tie-breakers.

## Slide 8 - Class imbalance strategy

Class imbalance is handled at several levels.

First, the main anomaly models learn the normal pattern rather than depending on
a balanced target. Second, evaluation uses PR-AUC and F2 instead of accuracy.
Third, thresholds are tuned according to business cost rather than a default
cutoff.

I also train a supervised Random Forest benchmark. When imbalanced-learn is
available, SMOTE increases the minority share to five percent inside the training
pipeline only, and the forest also uses class weighting. Applying SMOTE before
the split would leak synthetic information into validation and test sets, so the
project specifically avoids that mistake.

## Slide 9 - Business-aware threshold

Each model produces a continuous suspicion score. I convert that score into an
alert using a threshold selected on validation data.

The objective is to minimize a relative cost equal to 25 times false negatives
plus one times false positives. This assumes that missing fraud is substantially
more costly than reviewing a legitimate alert.

The ratio is illustrative. In a bank, it should be replaced by actual fraud
loss, expected recovery, manual investigation expense, customer friction, and
risk appetite. By stating the ratio, the decision is transparent and can be
challenged or recalibrated.

## Slide 10 - Validation comparison

This slide shows the validation precision-recall and ROC curves.

The most important chart is the precision-recall curve because the positive
class is extremely rare. ROC-AUC can still look strong when the number of false
positives is operationally unacceptable.

On validation data, the leading anomaly detector was [RUN OUTPUT: best anomaly
model on validation]. Its validation F2 score was [RUN OUTPUT: validation F2]
and PR-AUC was [RUN OUTPUT: validation PR-AUC]. The supervised benchmark
achieved [RUN OUTPUT: supervised validation PR-AUC], which indicates [state
whether labels materially improved ranking].

All thresholds were fixed at this stage. I did not tune them on test outcomes.

## Slide 11 - Locked test results

This table reports the locked test results.

The selected anomaly detector achieved a PR-AUC of [RUN OUTPUT: test PR-AUC],
precision of [RUN OUTPUT: test precision], recall of [RUN OUTPUT: test recall],
and F2 score of [RUN OUTPUT: test F2]. Its false-positive rate was [RUN OUTPUT:
test FPR].

The IQR baseline achieved [RUN OUTPUT: IQR PR-AUC], demonstrating [state whether
multivariate methods improved on amount-only detection]. Local Outlier Factor
and PCA achieved [RUN OUTPUT: brief comparison]. The supervised benchmark
achieved [RUN OUTPUT: benchmark PR-AUC and recall].

The comparison must be interpreted jointly. A model with slightly higher
PR-AUC may still be less suitable if its selected operating point creates an
unmanageable alert queue.

## Slide 12 - Confusion matrix and operational impact

At the locked test threshold, the selected anomaly detector created [RUN OUTPUT:
alerts] alerts. It captured [RUN OUTPUT: true positives] fraudulent transactions,
missed [RUN OUTPUT: false negatives], and sent [RUN OUTPUT: false positives]
legitimate transactions for review.

Its estimated relative cost, under the 25-to-1 assumption, was [RUN OUTPUT:
relative cost]. Precision at the top 100 alerts was [RUN OUTPUT: precision at
100], which represents the yield when investigators have a fixed daily capacity.

This operational view is more useful than quoting one abstract model score. It
connects model output with investigation workload and fraud-control benefit.

## Slide 13 - Deployment

The notebook packages the strongest deployable anomaly detector together with
its RobustScaler, feature order, locked threshold, and metadata using joblib.

A Flask service exposes two endpoints. The health endpoint reports whether the
model bundle is loaded. The prediction endpoint validates the transaction,
performs the same feature engineering and scaling used during training,
calculates the anomaly score, and returns either allow or manual review.

The project includes a Dockerfile and Gunicorn command for portable deployment.
In production, I would add authentication, encryption, rate limiting, centralized
logging, model registry controls, secret management, and high-availability
infrastructure.

## Slide 14 - Monitoring, ethics, and limitations

Post-deployment monitoring must cover schema failures, null rates, feature and
score drift, alert rate, recall and precision on matured labels, fraud value
captured, false-positive customer friction, review turnaround time, latency, and
API errors.

The model is decision support, not proof of fraud. A trained investigator and
approved policy should determine any adverse action. For auditability, each case
should retain the model version, score, threshold, timestamp, and reviewer
outcome.

The benchmark has important limitations. It covers only two days, most features
are anonymized, and merchant, device, geographic, and network relationships are
absent. Protected demographic fields are also unavailable, so demographic
fairness cannot be measured. Their absence is not evidence that the model is
fair.

## Slide 15 - Conclusion and next steps

I conclude with three points.

First, class imbalance changes both model evaluation and operating decisions;
accuracy is not a meaningful fraud-detection objective.

Second, comparing statistical, isolation-based, local-density, and reconstruction
methods provides a stronger conclusion than using one preferred algorithm.

Third, deployment quality depends on the threshold, monitoring, auditability,
and human-review process as much as on the model itself.

Future work would include rolling temporal validation, calibrated supervised
models, autoencoders, graph-based fraud detection, richer entity and device
features, explanation workflows, and live drift monitoring.

Thank you. I am ready for questions.
