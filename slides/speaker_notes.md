# Speaker notes

## Slide 1: How Much Personal History Is Enough?
"Thank you for having me. I'm Briyan Dyju, an independent researcher in Dubai, and I want to ask a simple question: how much of a person's own history does an anomaly detector need?"
[Time: 0:15]
-> Transition: "Let me start with a customer on their first day."

## Slide 2: A new customer opens an account
"A personalised detector compares each payment with your own past behaviour. That's powerful, because a 900-unit purchase can be ordinary for one person and extraordinary for another. But on day one, there is no past to compare with."
[Time: 1:15]
-> Transition: "So when does the personal baseline start to work?"

## Slide 3: Where does the baseline start to work?
"Production systems handle this with a cutoff. One patent family literally filters out 'unseasoned users'. But we could not find any published measurement of where that cutoff should be, so that is what I measured."
[Time: 0:45]
-> Transition: "Here is how I measured it."

## Slide 4: Measure it with one fixed test window
"For every account I hold the last sixty transactions fixed as a test window, and train on the n transactions just before it. To label anomalies without circularity, I swap in a real transaction from a different account, so provenance defines the label, not any detector."
[Time: 1:15]
-> Transition: "Here is what I ran this on."

## Slide 5: Three data layers, six detectors
"Three layers: real bank accounts from the Berka dataset, a controlled simulation where I set how variable people are, and a second, synthetic card dataset. Six standard unsupervised detectors run on each layer."
[Time: 0:45]
-> Transition: "Now the headline result."

## Slide 6: 25 transactions reach 90% of best
"This is the data-efficiency curve on real bank accounts. The shrinkage Mahalanobis detector reaches ninety percent of its best precision-recall AUC, 0.657, at just twenty-five personal transactions, and has gained almost everything by fifty. The two simple z-scores never learn at all."
[Time: 1:30]
-> Transition: "But that single number depends on how you define sufficiency."

## Slide 7: The threshold depends on the definition
"Rel-90 is the smallest n that reaches ninety percent of the best value; Rel-95 uses ninety-five. For Mahalanobis it is twenty-five either way, and it is identical across five independent cohort draws. But the stability criterion, spread across people, is never met by any detector that actually learns."
[Time: 0:45]
-> Transition: "Now a question that matters in practice: should you personalise at all when history is thin?"

## Slide 8: Below n = 10, trust the crowd
"The green line is a model built only from four hundred other accounts. Below about ten transactions, that crowd model beats the person's own history. So personalisation is not a safe default in the sparse regime."
[Time: 1:30]
-> Transition: "The orange line shows a fix."

## Slide 9: Shrink toward the crowd
"The hybrid shrinks the personal estimate toward the population estimate with a weight set only by n. At zero history it is exactly the crowd model, and as n grows it converges to the personal one. In my simulation it fails when the personal signal lives in the features themselves, so it is not a universal cure."
[Time: 0:45]
-> Transition: "Now the part that surprised me most: the alert threshold."

## Slide 10: How you set the alert threshold matters
"A detector needs an alert threshold. If you set it from the model's own training scores, at ten transactions it flags ninety-six percent of everything. With leave-one-out calibration, the same model flags one point two percent. That is an artefact of in-sample scoring, not of data scarcity."
[Time: 1:30]
-> Transition: "Does any of this hold on a different dataset?"

## Slide 11: A second dataset reverses a ranking
"On the synthetic card dataset the curve has the same shape, but the threshold moves to two hundred and fifty. And isolation forest, the worst detector on Berka, is the best one at small n here. So which model wins under scarcity depends on the dataset."
[Time: 1:15]
-> Transition: "What makes some people need more history than others?"

## Slide 12: Variability sets how fast it learns
"In the simulation, where I control variability, regular people need about fifty transactions and highly variable people about five hundred. But in the real Berka accounts, none of the eight descriptors I tested predicts an account's requirement. The simulation shows the effect can exist, not that it does in real data."
[Time: 1:00]
-> Transition: "One more practical finding about which history to keep."

## Slide 13: Recent history beats more history
"Going from fifty to two hundred and fifty transactions changes almost nothing, while drawing the same fifty from a four-times-longer span costs a little. I tested the obvious drift explanation directly, and it does not explain which accounts lose."
[Time: 0:45]
-> Transition: "Before I wrap up, what this study cannot tell you."

## Slide 14: Limits and ethics
"This is one bank in the nineties, plus a synthetic dataset, and my injected anomalies are far more common than real fraud. A flagged transaction is a deviation, not evidence of fraud. My fairness audit found no district-level gap, but it cannot test age, tenure or immigration status."
[Time: 0:45]
-> Transition: "So what should you take away?"

## Slide 15: Takeaways
"To summarise: a personal baseline ranks well from about twenty-five transactions; a shrinkage hybrid removes the cold-start penalty; how you calibrate the alert threshold changes the operating point; and recent history is enough."
[Time: 0:45]
-> Transition: "I'm happy to take questions."

## Slide 16: Questions?
"Thank you. The paper is linked at the QR code and I'm happy to take questions."
[Time: 0:15]
