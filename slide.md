# Real-Time Financial Fraud Detection Platform: 10-minute talk script

Speaker script for the 10-minute final presentation (problem → design → challenges → results → next steps). One section per slide; the timings add up to ~10 minutes. The same text is in each slide's speaker notes in the deck.

## Slide 1: Title  _(~30 s, ends at 0:30)_

Hi, I'm dp. Over the next ten minutes I'll walk you through the fraud detection platform I built: the problem, how I designed it, what went wrong along the way, the results, and what I'd do next. In one sentence: it reads live transactions from Kafka, scores each one for fraud in about a millisecond, uses a graph neural network to find fraud rings, and gives analysts a dashboard to review the cases.

## Slide 2: The problem  _(~60 s, ends at 1:30)_

Card-not-present fraud has to be stopped while the payment is being authorised, so there are four hard constraints. First, latency: the decision has to come back in milliseconds, using only what the account has done up to that moment. Second, imbalance: only about one transaction in twenty-five is fraud, so a model that always says 'not fraud' is ninety-six percent accurate and completely useless. That's why I evaluate with precision-recall, never accuracy. Third, organised fraud: rings of mule accounts share devices and stolen cards, and each account on its own looks normal. Fourth, fraudsters adapt, so the system has to detect drift and retrain itself.

## Slide 3: Data  _(~50 s, ends at 2:20)_

For data I compared the five suggested datasets and chose the IEEE-CIS format. It's the only one with real entity columns: card, billing address, e-mail and device fingerprint. Those are exactly what you need to build a graph of accounts and devices. It also has a real time axis, which velocity features need. The run I'm showing used 303 thousand transactions over six months, with a four percent fraud rate. It includes fraud rings, account takeovers and some deliberately hard, low-signal fraud. EDA showed fraud is heavily concentrated at night and in one product category, and fraud amounts are about three times higher than legitimate ones.

## Slide 4: Architecture  _(~60 s, ends at 3:20)_

Here's the architecture. The top half is offline training: feature engineering writes into the Feast feature store, then XGBoost and the graph neural network train and are blended, and everything is logged in MLflow. The bottom half is online: a Kafka producer replays transactions, and the FastAPI service consumes them. It computes features, fetches profiles from Feast, scores, explains, opens a case if needed and publishes the result back to Kafka. Analysts work in the Streamlit dashboard. On the right, Evidently checks for drift every hour and can trigger retraining, and the API hot-reloads the new model without downtime. It all deploys on Kubernetes.

## Slide 5: Features and Feast  _(~60 s, ends at 4:20)_

The most important design decision was about features. The classic failure in fraud systems is training/serving skew: features computed one way in pandas for training and another way in the live service. So every feature has exactly one definition. Real-time features (velocity, amount versus the user's average, device and location change) are computed only from earlier events. Long-horizon profiles live in Feast as daily snapshots. Training reads them with a point-in-time join, so it never sees the future, and serving reads the same views from the online store. I didn't just assume parity: a test replays 8,000 events through both code paths and asserts they're identical.

## Slide 6: Models  _(~60 s, ends at 5:20)_

There are two models. XGBoost handles the tabular signal. I trained three variants (class weights, SMOTE and no weighting) and tracked all of them in MLflow, judged by PR-AUC on a time-ordered split. The second model is a GraphSAGE network in PyTorch Geometric, over a graph of accounts linked to the devices and cards they use. It's trained with the label trick: known fraud labels are fed in as inputs for half the accounts while the loss is computed on the other half. That way it learns guilt-by-association without ever seeing the label it's predicting. A logistic stacker fitted on the validation period blends the two, and the alert threshold is set for 80 percent precision.

## Slide 7: Streaming and serving  _(~50 s, ends at 6:10)_

Serving is a FastAPI service that also runs the Kafka consumer, so the HTTP API and the stream share exactly the same scoring code. My first version took 26 milliseconds per transaction because it built pandas data frames. I rewrote it as a pure-Python fast path with one Feast round trip, now 1.1 milliseconds at the median and 1.6 at p95, about 826 transactions per second per core. Every alert carries its top reasons, so an analyst sees why: for example, the amount is four times this account's average, or the device is shared by fourteen accounts.

## Slide 8: Monitoring and analysts  _(~50 s, ends at 7:00)_

Analysts work in a Streamlit dashboard: an alert queue, a case view showing why each transaction was flagged and related cases on the same account or device, and buttons to confirm fraud or mark it legitimate. Those decisions become feedback labels. A Kubernetes cron job runs an Evidently drift report every hour. It retrains automatically if more than 30 percent of features drift, if amount behaviour drifts on its own, or if live PR-AUC on the feedback labels drops by more than 0.10. The API notices the new model file and hot-reloads it without restarting.

## Slide 9: Results  _(~60 s, ends at 8:00)_

Now the results, on the last 15 percent of the timeline, which the model never saw. A random model scores a PR-AUC of 0.02, the base rate. The graph model alone, with no transaction features at all, reaches 0.754, which shows it really finds the rings. XGBoost gets 0.867 and the blend 0.871. At the threshold chosen on validation, the system raises 928 alerts, two percent of traffic, with 85.9 percent precision and 87.7 percent recall. So about six out of every seven alerts an analyst opens are real fraud.

## Slide 10: End-to-end proof  _(~40 s, ends at 8:40)_

Offline numbers aren't enough, so I ran the whole system. Streaming 5,000 held-out transactions through the live API caught 149 of 164 frauds, 90.9 percent recall, at 82.8 percent alert precision. On healthy traffic the drift monitor sees 12 percent drift and correctly doesn't retrain. When I injected a three-times shift in amounts, it detected the drift, retrained the whole pipeline and the API hot-swapped the model, all automatically. Eighteen tests guard all of this.

## Slide 11: Challenges  _(~70 s, ends at 9:50)_

Four challenges taught me the most. First, the graph network initially made the blend worse, 0.870 down to 0.862. When I analysed the misses, XGBoost's device-sharing counts were already catching most rings. So I added the label trick and a device-level ring score, which catches account takeovers from ring devices, plus a guard that drops the GNN if it doesn't help on validation. Second, class weighting didn't beat no weighting, because PR-AUC measures ranking, and weighting mostly changes calibration. I was honest about that and kept the baseline in MLflow. Third, the drift monitor reported 54 percent drift on healthy traffic, because counters like account age always grow. Excluding those brought it to 12 percent. Fourth, latency went from 26 milliseconds to 1.1.

## Slide 12: Next steps  _(~40 s, ends at 10:30)_

Next, I'd run the full Kaggle IEEE-CIS data with feature selection over its V-columns. I'd also handle label delay, since real chargebacks arrive weeks late, and make the graph model inductive so brand-new accounts are scored within minutes. On the systems side: move stream state to Redis to scale out, use a cost-based threshold that weighs amount against analyst capacity, and add champion/challenger rollout from the MLflow registry. Thank you, I'm happy to take questions.

---
Total ≈ 10 min 30 s. Keep ~30 s spare for the switch to Q&A.
