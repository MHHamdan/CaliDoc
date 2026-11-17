# CaliDoc: WACV 2027 Talk, Speaker Notes

Paper #1597, *CaliDoc: Calibrated Field-Level Risk and Annotation Auditing for Document Information Extraction*, Mohammed Hamdan and Mohamed Cheriet (Synchromedia Laboratory, ÉTS Montréal).

Main talk: 14 slides, 1735 spoken words, about 11:55 at 145 words per minute. Backup slides B1–B10 are for Q&A only. These notes are generated from `source/notes.js`, the same text embedded in the PPTX Notes pages.

Conventions: `[pause]`, `[slow]` and `[close]` are delivery cues and are not spoken. Every number is traced in `CaliDoc_WACV2027_Claim_Audit.md`.

## Timing at a glance

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Title | 0:20 | 0:20 |
| 2 | Extraction pipelines consume complete fields | 0:30 | 0:50 |
| 3 | 89.8% token coverage, 0 of 420 fields covered | 1:05 | 1:55 |
| 4 | Change the object being calibrated | 0:45 | 2:40 |
| 5 | CaliDoc framework | 0:45 | 3:25 |
| 6 | Span score and document-level CRC | 1:15 | 4:40 |
| 7 | Field-conditional thresholds with pooled fallback | 0:45 | 5:25 |
| 8 | Main results (Table 2, Figure 1a) | 1:05 | 6:30 |
| 9 | CORD trade-off | 0:55 | 7:25 |
| 10 | Boundary-escape test | 1:10 | 8:35 |
| 11 | Detected defects and repair validation | 1:00 | 9:35 |
| 12 | Closest prior work | 0:50 | 10:25 |
| 13 | Limitations | 0:45 | 11:10 |
| 14 | Takeaways and resources | 0:45 | 11:55 |

## Slide 1: Title

**Time:** 0:20 (cumulative 0:20)

**Say:**

> Good morning. I'm Mohammed Hamdan from the Synchromedia Laboratory at ÉTS Montréal; this is joint work with Mohamed Cheriet.
>
> When a document extractor comes with a conformal guarantee, what exactly is the guarantee about? It can be about the wrong object. Fixing the object also lets us audit the annotations.

- **Emphasize:** "What is the guarantee about?" Set this up as the question the whole talk answers.
- **Why it matters:** Frames the talk as a measurement question, which fits the Evaluations & Datasets track.
- **Transition:** Let me start with what an extraction pipeline actually consumes.
- **Avoid:** Don't open with "we propose a new conformal method". The contribution is the guaranteed object plus the audit, not new CRC theory.
- **Sources:** Main paper p.1 (title, authors, affiliation).

## Slide 2: Extraction pipelines consume complete fields

**Time:** 0:30 (cumulative 0:50)

**Say:**

> Extraction feeds expense systems, records and compliance checks, and those systems consume complete fields, such as a company name, an address or a total.
>
> A field is right only if every word in its span is right. One wrong word in an eleven-word address gives you a wrong address.
>
> Conformal prediction for key information extraction has so far certified tokens, or one representative token per entity. Does that proxy tell us anything about fields?

- **Emphasize:** "Every word in its span." This is the event we will eventually certify.
- **Why it matters:** Sets up the mismatch between the unit that is certified and the unit that is delivered.
- **Transition:** Here is what happens when we look at the field level.
- **Avoid:** The receipt and chip graphics on this slide are conceptual. Don't describe them as data.
- **Sources:** Main Sec. 1 (p.1); Sec. 2.2 (p.2).

## Slide 3: 89.8% token coverage, 0 of 420 fields covered

**Time:** 1:05 (cumulative 1:55)

**Say:**

> This is the result that started the project. On SROIE, a LayoutLMv3-Large tagger gets an entity F1 of about 94.5, so the extractor itself is good. We add standard token-level conformal prediction, with LAC scores and a 90% target. Token coverage comes out at 89.8%. By the usual criterion, calibration worked.
>
> Now count complete fields instead. There are 420 gold field spans in the evaluation documents, and not one of them has all of its tokens covered. The count is zero.
>
> [pause] Both numbers are correct, and they come from the same predictions.
>
> The reason is that a single marginal threshold is sized by the majority class. The O class makes up 84% of tokens, so the threshold, 0.0056, is set to serve O. Six of the nine label classes get zero coverage, including all four field-initial B- classes. Every field starts with a B- token, so no field can be fully covered. A token-level guarantee was never a statement about fields.

- **Emphasize:** Say the two numbers slowly: "eighty-nine point eight percent of tokens" and then "zero of four hundred twenty fields". The extractor is good, with F1 94.54.
- **Why it matters:** This is the counterexample the whole paper rests on. If the audience remembers one number, it should be 0/420.
- **Transition:** To see what changes if we calibrate fields instead, let's look at one real receipt.
- **Avoid:** Don't say token CP is "wrong" or "broken". It delivers exactly its marginal token guarantee. The point is that token coverage is the wrong unit for fields. Don't claim this happens on every dataset; it happens when a high-support class dominates marginal calibration.
- **Sources:** Main Sec. 1 (p.1), Sec. 4.4.2 and Table 1 (p.5), Sec. 4.3 (F1 94.54 ± 0.33, p.4); Supp. Table S6 (p.4).

## Slide 4: Change the object being calibrated

**Time:** 0:45 (cumulative 2:40)

**Say:**

> This is one real SROIE test receipt, scored by our released checkpoint. I'm showing its structure rather than its text, because SROIE's redistribution licence is unclear.
>
> The receipt has four fields. Each dot marks the score of that field's worst word. The red line is the token-marginal threshold. Every field's worst word lies beyond it, so all four fields are missed.
>
> The teal ticks are the field-level thresholds that CaliDoc calibrates. All four fields fall inside them.
>
> Nothing about the extractor changed. What changed is the object we calibrate: the complete field, with the document as the exchangeable unit. That is the core idea of CaliDoc.

- **Emphasize:** Worst word against threshold. One dot per field decides whether that field is fully covered.
- **Why it matters:** Makes the conceptual shift concrete on genuine model output before any equation appears.
- **Transition:** Here is how CaliDoc is organized end to end.
- **Avoid:** This is one receipt, shown for illustration. The aggregate evidence is Table 2. The marginal threshold is the Table 1 value (mean over three calibration splits); the field thresholds are the released first-split values (Supp. Table S13).
- **Sources:** assets/examples/sroie/sroie_test_X51005568889/prediction.json (released checkpoint MHamdan/calidoc-layoutlmv3-large-sroie); main Table 1 (q = 0.0056); Supp. Table S13 (λ_t: ADDRESS 0.1013, COMPANY 0.1402, DATE 0.0224, TOTAL 0.0765).

## Slide 5: CaliDoc framework

**Time:** 0:45 (cumulative 3:25)

**Say:**

> CaliDoc runs entirely on top of a fixed extractor. We take LayoutLMv3-Large with its ordinary argmax decoding, fit a temperature on a held-out early-stopping split, and score every field span by its worst word.
>
> From there, two paths share one calibrated threshold.
>
> The calibration path runs conformal risk control over documents to choose thresholds. For each extracted field it returns a calibrated candidate set.
>
> The audit path reuses the same threshold as a reference, with no new free parameter, and asks where inside a field the escapes land. Is it at the boundaries, or anywhere? It answers that against an explicit null hypothesis.
>
> There's no retraining, no constrained decoding, and no sequence search.

- **Emphasize:** One threshold, two uses. Nothing upstream is retrained.
- **Why it matters:** Gives the audience a map for the rest of the talk: method, then guarantee, then audit.
- **Transition:** Let's start with the calibration path and with exactly what it guarantees.
- **Avoid:** Don't imply the audit runs per document. It is a corpus-level test of a field type.
- **Sources:** Main Sec. 3 (pp.2–4); temperature on early-stopping split: Sec. 3.1; Supp. S1.

## Slide 6: Span score and document-level CRC

**Time:** 1:15 (cumulative 4:40)

**Say:**

> Two design decisions.
>
> First, the score. A span is fully covered at a threshold lambda exactly when its worst word is below lambda. So we score a span by its maximum word score, and full-span coverage becomes a one-dimensional problem. That choice is forced: a mean or a quantile would certify a different event. The price is conservatism on long spans.
>
> Second, the unit. Spans on one page aren't exchangeable, because one misaligned OCR line corrupts several of them. So the document is the exchangeable unit, and a document's loss is the fraction of its gold fields that are missed. That loss is monotone and bounded, so conformal risk control applies: take the smallest lambda whose corrected calibration risk is at most alpha.
>
> The theorem: on a fresh exchangeable document with at least one gold field, the expected fraction of missed gold fields is at most alpha.
>
> [slow] That's an expectation. It doesn't promise that any single document misses at most ten percent, or that each field type is covered at ninety percent. The full-span coverage numbers I'll show next are empirical, not the theorem.

- **Emphasize:** "Expected fraction of missed fields per document." Name the quantity explicitly, then list what it is not.
- **Why it matters:** A conformal-prediction expert will judge the paper on whether the guaranteed quantity, the exchangeable unit, and the assumptions are stated precisely.
- **Transition:** That was the pooled threshold. Field types differ a lot, which brings us to field-conditional thresholds.
- **Avoid:** Never say "guarantees 90% field coverage". The assumptions are exchangeability of calibration and test documents, and calibration documents disjoint from training, temperature fitting, and model selection.
- **Sources:** Main Sec. 3.3, Eqs. 1–3 and the guarantee statement (p.3); Sec. 5.2 (p.8); Supp. S4 (p.3).

## Slide 7: Field-conditional thresholds with pooled fallback

**Time:** 0:45 (cumulative 5:25)

**Say:**

> Running the same risk control per field type gives each type its own threshold. A label enters a word's prediction set when its score is under its own type's threshold, so the rule we deploy and the event we calibrate coincide.
>
> The catch is support. Types with fewer than twenty-five calibration spans fall back to the pooled threshold; otherwise their threshold degenerates and the set becomes the whole label space. On SROIE, all four types are supported. On CORD, most are not.
>
> So, precisely: the pooled bound holds, and each supported type has its own bound. Fallback types inherit only the pooled bound, and aggregate coverage under this mixed rule is empirical.

- **Emphasize:** "Fallback types inherit only the pooled bound."
- **Why it matters:** Prevents the most likely over-reading, which is a uniform per-field guarantee.
- **Transition:** With that precision in place, here are the results across all five settings.
- **Avoid:** Don't call the mixed rule "field-conditional coverage for every field". The m0 sweep is descriptive, not tuning.
- **Sources:** Main Sec. 3.3 'Field-conditional thresholds' (p.3); Supp. S4 (p.3), Table S14 (p.8): 9 supported of 24 listed CORD types; 19.3% of evaluation spans in fallback types (Supp. p.5). Supp. S4.3 prose says '16 of 25'; see Claim Audit discrepancy D1.

## Slide 8: Main results (Table 2, Figure 1a)

**Time:** 1:05 (cumulative 6:30)

**Say:**

> Here is the whole picture at the ninety percent target. The bars show empirical full-span coverage.
>
> On SROIE, the token-marginal threshold leaves a realized field-miss risk of 1.000, which means every gold field is missed on every document. Field-conditional SpanCRC brings that to 0.091, against the 0.10 bound on its expectation. Empirically, full-span coverage goes from zero to 0.909 and span CovGap falls from 90 points to 2.1, for sets about ten percent larger.
>
> Repaired SROIE, which I'll come back to, behaves the same way.
>
> On WildReceipt and FUNSD, the marginal threshold was already working, and SpanCRC leaves them roughly where they were. WildReceipt is close to free; FUNSD pays 23 percent in set size for a modest CovGap gain. We don't claim universal improvement: full-span coverage on those two datasets moves down slightly, by half a point or so.
>
> The method helps in proportion to how badly the marginal threshold fails. Then there's CORD.

- **Emphasize:** SROIE: risk 1.000 → 0.091, coverage 0.000 → 0.909, CovGap 90.0 → 2.1 pp, about +10% set size.
- **Why it matters:** This is the main quantitative evidence. It shows a large repair where marginal calibration fails and nothing breaks where it works.
- **Transition:** CORD is our weakest case, so let's look at it directly.
- **Avoid:** Only the risk columns relate to the theorem; coverage, CovGap and set size are empirical. Don't read the table row by row. Set-size changes are from Supp. Fig. S1.
- **Sources:** Main Table 2 and Fig. 1(a) (p.6), Sec. 4.5 (p.5); Supp. Fig. S1 (set-size %), Table S3 (SROIE 0.90 → 0.99).

## Slide 9: CORD trade-off

**Time:** 0:55 (cumulative 7:25)

**Say:**

> CORD is where we pay. Most field types fall back, so span CovGap improves only from 42.3 to 33.3 points, while average set size more than triples, from 0.90 to 2.92. Thinly supported types get very large thresholds, so their labels enter most sets. Realized risk on its single split is 0.108, just above alpha. That's consistent with a bound on an expectation, but you should see it.
>
> There is a repair that keeps a guarantee. Fix the threshold shape on one half of the calibration documents and choose one level by CRC on the other half. Set size drops to 1.12 and CovGap to 18.4 points. But that keeps only the pooled document-level bound, not a field-conditional one, and it needs enough calibration documents; on FUNSD it inflates sets.

- **Emphasize:** Show the cost plainly: 0.90 → 2.92. Then give the fix with its narrower guarantee.
- **Why it matters:** Disclosing the weakest result before anyone asks builds credibility with expert reviewers.
- **Transition:** Now the second use of the same calibrated threshold: auditing annotations.
- **Avoid:** Never present level-calibrated interpolation as a field-conditional bound (Supp. Prop. 3 is pooled only).
- **Sources:** Main Sec. 4.5 (pp.5), Sec. 5.2 (p.8), Conclusion limitation 2 (p.8); Supp. S4.3, Proposition 3 and Table S12 (pp.3–5), S8.1 (p.10), Table S14.

## Slide 10: Boundary-escape test

**Time:** 1:10 (cumulative 8:35)

**Say:**

> Take the pooled threshold as a reference. A word escapes if its gold tag scores beyond it. In spans where some words escape but not all, we ask whether escapes concentrate on the two boundary words. That is the signature of alignment and BIO errors, where a span starts or ends one word off.
>
> The null is a conditional randomization test. Given each span's length and number of escapes, escaped positions are uniform within the span and independent across spans. Each span's edge count is then hypergeometric, and we get a Monte-Carlo p-value from 5,000 draws. A field that is simply hard escapes everywhere and shows no enrichment.
>
> Two caveats. Cross-span independence is an assumption, since spans on one document can share a defect. So every p-value is paired with a document-blocked bootstrap interval, and we only claim a finding when the two agree. Holm handles multiplicity across fields, but it can't repair that assumption. And spans of two words or fewer carry no positional information.

- **Emphasize:** "Where inside the span", conditional on how many words escaped. Then name both caveats.
- **Why it matters:** The audit is what separates CaliDoc from a pure calibration paper, and its validity conditions must be explicit.
- **Transition:** Here is what the test finds, and how we checked it.
- **Avoid:** Don't call this a general label-error detector, and don't claim finite-sample validity under arbitrary within-document dependence. Ineligible fields are untested, not clean.
- **Sources:** Main Sec. 3.4, Eq. 4 (pp.3–4); Supp. S5 (pp.5–10), Propositions 1–2.

## Slide 11: Detected defects and repair validation

**Time:** 1:00 (cumulative 9:35)

**Say:**

> At familywise level 0.10, the test rejects two of SROIE's four eligible fields. ADDRESS shows edge enrichment of 4.07 and COMPANY 1.80, both with a Holm-adjusted p of 0.0008. Their document-blocked intervals exclude one. On the clean benchmarks, none of sixteen eligible field types is rejected, and Holm, BH and BY agree on every split. CORD's SERVICE_PRICE reaches 2.0, but on two informative spans, so the null is wide and nothing is rejected.
>
> Over escaped ADDRESS words, the most frequent prediction is B-ADDRESS at span edges: a one-word boundary offset.
>
> Then the falsification test. SROIE's official Task-3 release ships the gold field strings, so we re-derived the labels independently and evaluated them as a separate lineage. ADDRESS enrichment falls to 0.79, with p = 0.87, and COMPANY to 1.15. Nothing is rejected anywhere.
>
> Calibration didn't repair the data. An independent re-alignment did, and the audit registered it.

- **Emphasize:** ADDRESS 4.07 → 0.79 after independent repair. Say clearly that the repair is not CaliDoc.
- **Why it matters:** Positive detection plus a negative after repair is the strongest evidence the audit measures something real.
- **Transition:** How does this relate to the closest prior work?
- **Avoid:** The repaired results come from a separately evaluated annotation lineage; the original gate still failed (94.45% < 95%). The positive real-data validation is one dataset. ADDRESS and COMPANY share the same receipts.
- **Sources:** Main Sec. 4.6 and Fig. 1(b) (p.6), Sec. 4.9 (p.7); Supp. Table S17 (p.11), Table S18 (p.12), S6 (p.10).

## Slide 12: Closest prior work

**Time:** 0:50 (cumulative 10:25)

**Say:**

> The closest prior work is Rombach and Mehdiyev, published in IJDAR in 2026. They brought split conformal prediction to key information extraction, scoring each entity by its first token, with marginal coverage over entity instances. So conformal KIE isn't new, and we don't claim it is.
>
> What we change is the guaranteed object: the complete field, as a document-level risk. Mondrian and clustered CP fix class imbalance, but they still certify tokens.
>
> At their operating point on WildReceipt, our probabilities give similar coverage with smaller sets. We state that comparison rather than rank it, because backbone, score, schema and splits differ. The more telling number: on SROIE, the first-token unit reaches 0.897 while full-span coverage is 0.856.

- **Emphasize:** "Same tool, different guaranteed object." Name the prior work first, and generously.
- **Why it matters:** Reviewers will ask about novelty. The answer is the unit of the guarantee plus the audit, not "first CP for KIE".
- **Transition:** Here are the limits of what we've shown.
- **Avoid:** No "first", no SOTA, no head-to-head ranking. The proxy-to-field gap depends on the dataset and the operating point.
- **Sources:** Main Sec. 2.2 (p.2), Sec. 4.4.1 (pp.4–5); Supp. Table S9 (p.6).

## Slide 13: Limitations

**Time:** 0:45 (cumulative 11:10)

**Say:**

> Six limitations, all of them in the paper.
>
> First, the guarantee is a risk, not a coverage, and a single split can exceed alpha, as CORD's does.
>
> Second, field-conditional calibration is limited by support, and CORD shows the cost.
>
> Third, the audit is blind on spans of two words or fewer, which leaves short-span schemas like WildReceipt largely untested.
>
> Fourth, the null assumes independence across spans.
>
> Fifth, the real-data positive rests on one corpus, SROIE, and both flagged fields share the same receipts.
>
> Sixth, the scope is four English receipt and form corpora and one backbone family, at two scales. Multilingual, long-document, and OCR-free settings are untested.

- **Emphasize:** Deliver this calmly and completely. It is a strength slide, not an apology.
- **Why it matters:** Pre-empts the hardest questions and shows that the claims are scoped.
- **Transition:** So what should you take away?
- **Avoid:** Don't add limitations that aren't in the paper, and don't soften the ones that are.
- **Sources:** Main Sec. 6 'Limitations' (p.8), Sec. 3.4 (p.4), Sec. 4.6 (p.6); Supp. S2.1 (Base backbone).

## Slide 14: Takeaways and resources

**Time:** 0:45 (cumulative 11:55)

**Say:**

> Three things to take away.
>
> Token-level reliability is not complete-field reliability. On SROIE, 89.8% of tokens were covered and zero of 420 fields were.
>
> Make the delivered field the unit. Conformal risk control over documents bounds the expected fraction of missed fields, without retraining the extractor.
>
> And use the same calibration to audit the annotations. A positional test with an explicit null localized a real annotation defect, and an independent repair silenced it.
>
> Code, released checkpoints, the supplement, and a Colab demo are linked here. This work was funded by an NSERC Discovery Grant.
>
> [close] Report reliability for the object you actually deliver. Thank you, and I'm happy to take questions.

- **Emphasize:** Closing line: "Report reliability for the object you actually deliver."
- **Why it matters:** A memorable line tied directly to the contribution.
- **Transition:** Q&A. Backup slides B1–B10 follow.
- **Avoid:** Don't introduce new numbers here.
- **Sources:** Main Sec. 6 (p.8), Acknowledgment (p.8); Supp. Table S40 (p.21).

## Backup divider (backup)

**Say:**

> Backup slides, used only in Q&A.


## B1 · Mondrian on SROIE (backup)

**Say:**

> This answers "why not just use class-conditional CP?" On SROIE, Mondrian calibration repairs per-class token coverage cheaply. Every class returns to between 87 and 100 percent, at average set size 0.90, and CovGap falls from 71.1 to 3.4 points.
>
> Two things to note. First, it still certifies tokens, not fields. Second, the price is a quantile spread of a factor of 367, from 0.0027 for O to 0.9996 for I-TOTAL. That spread is a screening signal, not evidence, because part of it is thin support. That's why the audit uses a statistic with a null.

- **Emphasize:** Mondrian fixes per-class token coverage; it is still a token guarantee.
- **Why it matters:** Distinguishes CaliDoc from class-conditional CP.
- **Avoid:** Don't present the quantile spread as a label-noise test.
- **Sources:** Main Table 1 and Sec. 4.4.2 (p.5).

## B2 · Why the max score (backup)

**Say:**

> If you calibrate the pooled CRC threshold on other span scores, each lands near nominal on its own event but not on the delivered one. On SROIE, the mean score achieves 0.898 on its own event, but true full-span coverage is only 0.762 and realized document risk is 0.238, compared with 0.114 under the max.
>
> The gap grows with span length; length and span score correlate at Spearman 0.59 on SROIE against 0.04 on WildReceipt, where spans are at most two words long and all scores coincide.

- **Emphasize:** Other scores certify other events; only the max certifies complete-field coverage.
- **Why it matters:** Answers "why max and not mean?"
- **Avoid:** These are diagnostic ablations, not proposed alternatives.
- **Sources:** Main Sec. 3.3 (p.3), Sec. 4.5 (p.5); Supp. S4.1–S4.2, Tables S10–S11 (p.6).

## B3 · Small-calibration controls (backup)

**Say:**

> Could the per-class collapse just be a small-sample artifact? We ran three controls.
>
> First, a feasibility bound: a class-conditional quantile needs at least 8 calibration tokens at alpha 0.10, and classes below that are marked rather than scored.
>
> Second, a matched-n control on clean CORD classes: coverage is 0.810, 0.910, 0.954, 0.922, and 0.905 at n equal to 5, 10, 20, 50, and 100. That's a nine-point cost at n = 5, nowhere near a collapse to zero.
>
> Third, stronger baselines on CORD: clustered CP degenerates toward marginal, 45.9 versus 45.8 points; interpolation gets 32.2; full class-conditional gets 16.4 but grows set size from 0.90 to 2.49.

- **Emphasize:** Small n costs about nine points, not one hundred.
- **Why it matters:** Answers the calibration-size objection.
- **Sources:** Main Sec. 4.7 (p.7); Supp. Tables S28–S29.

## B4 · Audit size and power (backup)

**Say:**

> Size: when we redraw escape positions uniformly over 200 replicates, the familywise rejection rate under Holm is 0.025 on FUNSD and zero on WildReceipt and CORD, against 0.140 uncorrected on FUNSD.
>
> Power, under end-extension corruption: a single corrupted FUNSD QUESTION span is detected at rate 0.53 over 30 seeds, two spans at 0.73, and twenty at 0.93. WildReceipt never exceeds zero, which is the short-span blind spot. Start-side shifts reach at most 0.20.
>
> One disclosure: FUNSD QUESTION has a borderline signal on clean labels, raw p 0.036 and Holm 0.109, which is not a finding.
>
> Adding a pre-specified interior mask and correcting over all 36 field-mask pairs still rejects exactly the two SROIE fields, and only under the edge mask.

- **Emphasize:** Holm is doing visible work; the blind spot is structural.
- **Why it matters:** Answers "how do you know the test is calibrated and has power?"
- **Avoid:** Synthetic size and power are on our own probabilities. They are not a benchmark against other detectors.
- **Sources:** Main Sec. 4.6 'Size and power' (pp.6–7); Supp. Tables S16, S21–S23.

## B5 · Label-error detectors (backup)

**Say:**

> Under planted CORD noise, our coverage-deficit diagnostic is weaker than dedicated detectors at token-level ranking, and we say so directly. AUPRC is 0.007 to 0.169 across six noise conditions, against 0.047 to 0.951 for Cleanlab and 0.117 to 0.859 for AUM. Scoring a class-level statistic per token is the wrong use of it.
>
> At class level, under class-targeted noise, it ranks the corrupted class 6th to 12th of 54, against 21st to 46th for the baselines, but never in the top 3, and under uniform or BIO-shift noise it ranks 48th and 38th. Localization comes from the boundary-escape test, which conditions on position and carries a null.

- **Emphasize:** Different job: a field-level hypothesis test, not a token ranker.
- **Why it matters:** Answers "do you beat Cleanlab?" No, and that isn't the claim.
- **Avoid:** Don't claim superiority over label-error detectors.
- **Sources:** Main Sec. 4.8 (p.7); Supp. S7, Table S26 (p.17).

## B6 · Generality (backup)

**Say:**

> LayoutLMv3-Base under the identical protocol reproduces the SROIE collapse on all three seeds. Full-span coverage is 0.000, repaired to 0.901, and risk is 1.000, repaired to 0.099. The audit rejects exactly ADDRESS and COMPANY on all three seeds, and the clean-benchmark pattern matches. The one difference is that Base pays more set size on WildReceipt, 1.71 against 0.93.
>
> With a clean calibration protocol, retraining WildReceipt with calibration carved from the training partition and the official test set used only for final evaluation, F1 is 94.05 against 94.38 and every conformal conclusion replicates.

- **Emphasize:** Two scales, one family. Broader architectures are untested.
- **Why it matters:** Answers backbone and protocol questions.
- **Avoid:** Don't claim architecture-general results; OCR-free models are untested.
- **Sources:** Main Sec. 4.5 last paragraph (p.5); Supp. S2.1, Table S3 (pp.2–4).

## B7 · SROIE alignment and repair lineage (backup)

**Say:**

> The pre-registered gate required at least 95% exact contiguous string-to-token alignment per split. It failed at 94.68% on train and 94.02% on test, with a pooled rate of 94.45%. The result was identical across three independent data lineages, including the official competition site, and that is why SROIE is a case study rather than a benchmark.
>
> Punctuation-insensitive matching recovers another 2.80% and numeric normalization another 0.13%, reaching 97.38%. Address fails most often: 85.6% exact on test, against 95.1 to 99.1% for the other fields. We rebuilt under the relaxed matchers as a separate lineage; the gate decision stands.
>
> At token level, the repair looks mixed: B-ADDRESS rises from 0 to 0.444 and I-ADDRESS falls from 0.959 to 0.261. At the span and position level it is a clean confirmation.

- **Emphasize:** The repair is a separate lineage; it does not retroactively pass the gate.
- **Why it matters:** Answers "how strong is the repair evidence?"
- **Avoid:** Don't say the repaired data passes the gate.
- **Sources:** Main Sec. 4.9 (p.7); Supp. S6, Tables S24–S25 (pp.10, 16).

## B8 · Reading α in deployment (backup)

**Say:**

> Alpha equal to 0.10 bounds the expected per-document fraction of missed gold fields. On SROIE, with four required fields per receipt, that means at most 0.4 expected missed fields per receipt, and no individual receipt is guaranteed.
>
> Choosing alpha is a costing exercise: the expected review burden is roughly alpha times fields per document times volume, weighed against the cost of a miss.
>
> Set shape matters. On CORD, 100% of tokens carry multi-label sets, so every field arrives with alternatives and goes to triage. SROIE and WildReceipt stay near-singleton, at 99.0% and 92.9%, and there the cost is empty sets, 6.3% of WildReceipt tokens, which are forced abstentions.

- **Emphasize:** A bound on an average, not on a receipt.
- **Why it matters:** Answers deployment questions.
- **Avoid:** Per-field or adaptive alpha would need a guarantee we do not have.
- **Sources:** Main Sec. 5.2 (p.8); Supp. S8.1 (p.10).

## B9 · CaliDoc in action (CORD) (backup)

**Say:**

> This is the released CORD checkpoint on a CORD v2 test receipt, licensed CC BY 4.0. The extractor returns 25 fields and 24 match the gold labels. Because CORD sets are multi-label, all 25 fields are routed to review.
>
> The one incorrect extraction, Coupon 100,000, is predicted as SUB_TOTAL.ETC where the gold label is TOTAL.TOTAL_ETC. Its worst word scores 0.80 against the pooled threshold 0.189, so its calibrated set excludes the prediction. It is the only field in the conflict state.
>
> This is one illustrative case, not a sensitivity or specificity result, and CaliDoc does not guarantee that incorrect fields are detected.

- **Emphasize:** Routing is operational, not a correctness probability.
- **Why it matters:** Shows the released software on a licence-cleared document.
- **Avoid:** Don't present routing states as additional guarantees, and don't use this one receipt as an accuracy claim.
- **Sources:** Supp. S12 (pp.20–21); assets/examples/cord/cord_test_0014 (CC BY 4.0).

## B10 · Calibration and robustness (backup)

**Say:**

> After temperature scaling, ECE over three seeds is 1.27% on CORD, 1.81% on WildReceipt, and 6.37% on FUNSD.
>
> Under nine image perturbations at three severities, with thresholds held at their clean values, mean F1 drop is 0.56 points on FUNSD and mean coverage drop is below 0.15 points everywhere, with a worst cell of 1.53. This coverage is empirical, because perturbation breaks exchangeability.
>
> MC Dropout hurts: CORD ECE goes from 1.57 to 2.77% and coverage from 87.9 to 78.6%, at twenty times the inference cost.
>
> On CORD, 9.9% of tokens get an empty LAC set, and 68.2% of those would have been classified correctly.

- **Emphasize:** Coverage under shift is empirical, not guaranteed.
- **Why it matters:** Answers distribution-shift and baseline questions.
- **Avoid:** No guarantee under distribution shift.
- **Sources:** Main Sec. 4.4 (p.4), Sec. 4.10 (p.7); Supp. S8–S9.
