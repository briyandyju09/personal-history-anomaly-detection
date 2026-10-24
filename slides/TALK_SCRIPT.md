# Talk script: How Much Personal History Is Enough?

**Speaker**: Briyan George Dyju, Independent Researcher, Dubai, United Arab Emirates
**Talk type**: oral (15 min)
**Total slides**: 16

---

## Slide 1: How Much Personal History Is Enough? [0:00 - 0:15]

"Thank you, and good morning. I'm Briyan Dyju, an independent researcher based in Dubai. Today's talk asks one simple question: how much of a person's own transaction history does an anomaly detector need before it is worth using?"

-> *Transition*: "Let me start with a customer on their first day."

---

## Slide 2: A new customer opens an account [0:15 - 1:30]

"Imagine you open a bank account today. A personalised anomaly detector would compare each of your payments with your own past behaviour. That is a powerful idea: a 900-unit purchase is unremarkable for one account holder and extraordinary for another, and a single population-wide rule serves both of them badly. But the approach has an obvious dependency. The personal model has to be estimated from your own transactions, and when your account opens, there are none."

-> *Transition*: "So when does the personal baseline start to work?"

---

## Slide 3: Where does the baseline start to work? [1:30 - 2:15]

"Production systems deal with this using a cutoff. One patent family describes filtering out what it calls unseasoned users, meaning users with insufficient history. The cutoff is treated as a known engineering parameter. Yet in a documented search, we found no published measurement of where it lies, or even whether it exists. So the question for this talk is: how many personal transactions are enough?"

-> *Transition*: "Here is how I measured it."

---

## Slide 4: Measure it with one fixed test window [2:15 - 3:30]

"The design is simple. For every account, I hold the last sixty transactions fixed as the test window, and I train on the n transactions immediately before that window, with n running from one up to two hundred and fifty. Because the window never changes, every comparison across n is paired within the account. For ground truth, I avoid a trap: if you label anomalies with a statistical rule and then evaluate detectors that compute something similar, you only measure agreement between two versions of one rule. So I replace a held-out transaction with a real transaction from a different account. Provenance defines the label, and no detector's score does."

-> *Transition*: "Here is what I ran this on."

---

## Slide 5: Three data layers, six detectors [3:30 - 4:15]

"There are three data layers. The main one is Berka: real, anonymised retail banking from a Czech bank, with three hundred accounts evaluated. Because real data cannot give me controlled variability, I also simulate two hundred and forty people whose behaviour I set directly. And to test whether the shape is specific to one old dataset, I use IBM's TabFormer card data, which is synthetic and never pooled with the real results. Six standard unsupervised detectors run on each layer: Mahalanobis distance, k-nearest-neighbour distance, isolation forest, a small autoencoder, and two z-scores."

-> *Transition*: "Now the headline result."

---

## Slide 6: 25 transactions reach 90% of best [4:15 - 5:45]

"This is the main result, on real bank accounts. The horizontal axis is how many personal transactions the detector has, on a log scale, and the vertical axis is precision-recall AUC, where the dotted line is the 0.10 prevalence baseline. The Mahalanobis detector rises steeply and reaches ninety percent of its best value, 0.657, at twenty-five transactions. By fifty, ninety-eight percent of its total gain has arrived. The k-NN detector jumps later, between twenty-five and fifty, isolation forest is still climbing at two hundred and fifty, and the two univariate z-scores never learn anything."

-> *Transition*: "But that single number depends on how you define sufficiency."

---

## Slide 7: The threshold depends on the definition [5:45 - 6:30]

"I define sufficiency in three ways and report all of them. The relative-90 threshold is the smallest n that reaches ninety percent of the best observed value and never falls back. For Mahalanobis and the autoencoder that is twenty-five; for k-NN it is a hundred; for isolation forest it is still two hundred and fifty. The twenty-five for Mahalanobis is identical across five independent draws of the account cohort, so it is not an accident of one sample. The third definition, stability across people, is never met by any detector that learns, which tells us the average curve hides large individual variation that does not shrink with more data."

-> *Transition*: "Now a question that matters in practice: should you personalise at all when history is thin?"

---

## Slide 8: Below n = 10, trust the crowd [6:30 - 8:00]

"Here the green dashed line is a population model, built from four hundred other accounts and using none of this person's data. The blue line is the personal model. For Mahalanobis, the personal model is actually worse than the crowd until about ten transactions, and the effect sizes are large. For k-NN the crossover is later, at fifty. In every one of the five cohort draws, the Mahalanobis crossover falls at exactly n equals ten. The common assumption that personalisation is a safe default is wrong in the sparse regime."

-> *Transition*: "The orange line shows a fix."

---

## Slide 9: Shrink toward the crowd [8:00 - 8:45]

"The fix is a shrinkage hybrid. The personal mean and scatter are pulled toward the population estimates with a weight determined only by the history size n, and a strength kappa that I fixed at twenty-five in advance, never tuning it on test data. At n equals zero this is exactly the global model, and as n grows it converges to the personal model. On the real data it is never meaningfully worse than either parent. One honest caveat: in my simulation, where the strongest features are personal novelty indicators computed against your own history, the hybrid computes them against the population and destroys the signal. So it helps when the personal signal lives in the parameters, not when it lives in the features."

-> *Transition*: "Now the part that surprised me most: the alert threshold."

---

## Slide 10: How you set the alert threshold matters [8:45 - 10:15]

"Ranking quality is not the same as a usable alert. A deployed detector needs a threshold, and the obvious choice is the ninety-fifth percentile of the scores the model gives its own training data. At ten transactions, that makes the Mahalanobis detector flag ninety-six percent of all transactions, even though its PR-AUC is more than four times the baseline. The reason is that in-sample scores are biased low by construction: a model scores the very points it was fitted to. If you instead score each training point with a model fitted on the others, leave-one-out, the same detector flags only one point two percent at ten transactions, and from twenty-five its false-positive rate is 0.066 against a nominal 0.05. So the answer to how much history is enough depends on how you calibrate, and the effect holds across six detectors and five seeds, though not always in the same direction."

-> *Transition*: "Does any of this hold on a different dataset?"

---

## Slide 11: A second dataset reverses a ranking [10:15 - 11:30]

"On the TabFormer card data, which is synthetic, the qualitative shape replicates: Mahalanobis rises steeply and reaches the highest ceiling, and the z-scores stay flat. The relative-90 threshold, however, is two hundred and fifty rather than twenty-five. And one result does not replicate. Isolation forest was the worst detector at every size on Berka, but here it is the best at n equals five through fifty. I report that divergence plainly: the claim that sophisticated models are worse under scarcity is dataset-dependent, not universal."

-> *Transition*: "What makes some people need more history than others?"

---

## Slide 12: Variability sets how fast it learns [11:30 - 12:30]

"In the simulation I set variability directly, and it matters a great deal: a regular person saturates around fifty transactions, while a traveller or a highly variable person needs about five hundred. Variability changes the rate of learning far more than the ceiling. But on real Berka accounts, none of eight variability descriptors predicts an account's data requirement after correction. The simulation shows the effect exists when the generating parameters are what theory says; it is not evidence that it exists in real data."

-> *Transition*: "One more practical finding about which history to keep."

---

## Slide 13: Recent history beats more history [12:30 - 13:15]

"Does it help to keep more history? Going from fifty to two hundred and fifty transactions at natural density changes PR-AUC from 0.647 to 0.642, essentially nothing. Drawing the same fifty transactions from a span four times longer costs 0.036. The tempting explanation is behavioural drift, so I measured drift directly. It is real: older behaviour does look different from recent behaviour. But an account's drift does not predict how much that account loses, with a correlation of minus 0.08. The mechanism remains open. A practical bonus is that shorter retention seems to cost nothing here."

-> *Transition*: "Before I wrap up, what this study cannot tell you."

---

## Slide 14: Limits and ethics [13:15 - 14:00]

"The limits are real. The real evidence is one bank, from 1993 to 1998. The second dataset is synthetic. My injected anomalies are ten percent of transactions, far above realistic fraud prevalence, and I do not model an adversary imitating the account holder. Ethically, the transactions a detector flags are by construction different from a person's past, which in real life means holidays, house moves and new jobs. A flag is a deviation, not fraud. My district-level audit found no significant concentration of false positives by salary, unemployment or region, but with three hundred accounts it cannot exclude weak effects, and it cannot test age, tenure or immigration status."

-> *Transition*: "So what should you take away?"

---

## Slide 15: Takeaways [14:00 - 14:45]

"To summarise. A personal baseline ranks well from about twenty-five transactions on the best real-data detector, and that number is stable across cohorts. A hybrid that shrinks toward the crowd while history is thin removes the cold-start penalty without costing anything later. How you calibrate the alert threshold changes the operating point dramatically, so a single number of transactions is never the whole answer. And recent history is enough. The practical recommendation is not a number: use a hybrid, keep recent history, and do not assume that a detector that ranks well can be deployed."

-> *Transition*: "I'm happy to take questions."

---

## Slide 16: Questions? [14:45 - 15:00]

"Thank you. The paper is linked from the QR code on screen, at briyan dot xyz slash research, and my email is on the slide. I'm happy to take questions."

---

## Time budget summary

| Slide | Topic | Duration | Cumulative |
|:-:|---|:-:|:-:|
| 1 | How Much Personal History Is Enough? | 0:15 | 0:15 |
| 2 | A new customer opens an account | 1:15 | 1:30 |
| 3 | Where does the baseline start to work? | 0:45 | 2:15 |
| 4 | Measure it with one fixed test window | 1:15 | 3:30 |
| 5 | Three data layers, six detectors | 0:45 | 4:15 |
| 6 | 25 transactions reach 90% of best | 1:30 | 5:45 |
| 7 | The threshold depends on the definition | 0:45 | 6:30 |
| 8 | Below n = 10, trust the crowd | 1:30 | 8:00 |
| 9 | Shrink toward the crowd | 0:45 | 8:45 |
| 10 | How you set the alert threshold matters | 1:30 | 10:15 |
| 11 | A second dataset reverses a ranking | 1:15 | 11:30 |
| 12 | Variability sets how fast it learns | 1:00 | 12:30 |
| 13 | Recent history beats more history | 0:45 | 13:15 |
| 14 | Limits and ethics | 0:45 | 14:00 |
| 15 | Takeaways | 0:45 | 14:45 |
| 16 | Questions? | 0:15 | 15:00 |

**Total**: 15:00 (target: 15:00)

## Anticipated Q&A

### Q1: How does this compare with just using the population model?
**A**: "Below about ten transactions the population model is better for Mahalanobis, and the personal model is better from ten onward: the paired difference is +0.111 at n = 10 and +0.327 at n = 100. The shrinkage hybrid sits at or above both across the grid, so it is the safer choice."

### Q2: What are the main limitations?
**A**: "The real evidence is one bank from 1993 to 1998 and the second dataset is synthetic. Injected anomalies are at 10%, far above fraud prevalence, and I do not model an adversary who imitates the account holder. No real fraud labels were evaluated."

### Q3: How expensive is each detector?
**A**: "At n = 250 the median fit time is 0.46 ms for Mahalanobis, 5.0 ms for k-NN, 22.9 ms for the autoencoder and 156.3 ms for isolation forest. The best real-data detector is also the cheapest."

### Q4: Does this transfer to modern card data?
**A**: "The qualitative shape replicates on the synthetic TabFormer data with a later threshold of 250, but that is synthetic. Whether it holds on modern real card payments is untested."

### Q5: What was the most surprising finding?
**A**: "That the headline 96% flagged at n = 10 came from calibrating on in-sample scores. Under leave-one-out calibration the detector flags 1.2%, so the failure at n = 10 is silence, not noise."

### Q6: How sensitive is the hybrid to kappa?
**A**: "I fixed kappa = 25 in advance and did not tune it, so I have not measured sensitivity. Tuning on this data would likely improve the hybrid but would invalidate the comparison."

### Q7: Does drift explain why older history hurts?
**A**: "I tested that directly. Drift is real, with a median of 0.69 of an account's own recent standard deviations, but per-account drift does not predict that account's loss (rho = -0.081, Holm p = 0.32). The mechanism is open."

### Q8: What is next?
**A**: "A finer history grid between 10 and 25, alternative out-of-sample threshold rules such as split-half and population-derived, a targeted evaluation on entities with generator-labelled fraud, and a deeper real dataset with merchant fields and individual-level demographics."
