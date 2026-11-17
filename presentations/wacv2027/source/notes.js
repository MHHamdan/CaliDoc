// Speaker notes: single source of truth for the PPTX notes pages and
// CaliDoc_WACV2027_Speaker_Notes.md. Every number is checked in
// CaliDoc_WACV2027_Claim_Audit.md against the camera-ready main paper
// (main.pdf, sha256 02a649c4...) and the 22-page supplement (01597_supp.pdf, sha256 048542cc...).
//
// Fields: id, title, time (seconds), say (spoken narration, paragraphs),
// emphasize, why, transition, avoid, sources.

module.exports = [
  {
    id: 1, title: "Title", time: 20,
    say: [
      "Good morning. I'm Mohammed Hamdan from the Synchromedia Laboratory at ÉTS Montréal; this is joint work with Mohamed Cheriet.",
      "When a document extractor comes with a conformal guarantee, what exactly is the guarantee about? It can be about the wrong object. Fixing the object also lets us audit the annotations."
    ],
    emphasize: "\"What is the guarantee about?\" Set this up as the question the whole talk answers.",
    why: "Frames the talk as a measurement question, which fits the Evaluations & Datasets track.",
    transition: "Let me start with what an extraction pipeline actually consumes.",
    avoid: "Don't open with \"we propose a new conformal method\". The contribution is the guaranteed object plus the audit, not new CRC theory.",
    sources: "Main paper p.1 (title, authors, affiliation)."
  },
  {
    id: 2, title: "Extraction pipelines consume complete fields", time: 30,
    say: [
      "Extraction feeds expense systems, records and compliance checks, and those systems consume complete fields, such as a company name, an address or a total.",
      "A field is right only if every word in its span is right. One wrong word in an eleven-word address gives you a wrong address.",
      "Conformal prediction for key information extraction has so far certified tokens, or one representative token per entity. Does that proxy tell us anything about fields?"
    ],
    emphasize: "\"Every word in its span.\" This is the event we will eventually certify.",
    why: "Sets up the mismatch between the unit that is certified and the unit that is delivered.",
    transition: "Here is what happens when we look at the field level.",
    avoid: "The receipt and chip graphics on this slide are conceptual. Don't describe them as data.",
    sources: "Main Sec. 1 (p.1); Sec. 2.2 (p.2)."
  },
  {
    id: 3, title: "89.8% token coverage, 0 of 420 fields covered", time: 65,
    say: [
      "This is the result that started the project. On SROIE, a LayoutLMv3-Large tagger gets an entity F1 of about 94.5, so the extractor itself is good. We add standard token-level conformal prediction, with LAC scores and a 90% target. Token coverage comes out at 89.8%. By the usual criterion, calibration worked.",
      "Now count complete fields instead. There are 420 gold field spans in the evaluation documents, and not one of them has all of its tokens covered. The count is zero.",
      "[pause] Both numbers are correct, and they come from the same predictions.",
      "The reason is that a single marginal threshold is sized by the majority class. The O class makes up 84% of tokens, so the threshold, 0.0056, is set to serve O. Six of the nine label classes get zero coverage, including all four field-initial B- classes. Every field starts with a B- token, so no field can be fully covered. A token-level guarantee was never a statement about fields."
    ],
    emphasize: "Say the two numbers slowly: \"eighty-nine point eight percent of tokens\" and then \"zero of four hundred twenty fields\". The extractor is good, with F1 94.54.",
    why: "This is the counterexample the whole paper rests on. If the audience remembers one number, it should be 0/420.",
    transition: "To see what changes if we calibrate fields instead, let's look at one real receipt.",
    avoid: "Don't say token CP is \"wrong\" or \"broken\". It delivers exactly its marginal token guarantee. The point is that token coverage is the wrong unit for fields. Don't claim this happens on every dataset; it happens when a high-support class dominates marginal calibration.",
    sources: "Main Sec. 1 (p.1), Sec. 4.4.2 and Table 1 (p.5), Sec. 4.3 (F1 94.54 ± 0.33, p.4); Supp. Table S6 (p.4)."
  },
  {
    id: 4, title: "Change the object being calibrated", time: 45,
    say: [
      "This is one real SROIE test receipt, scored by our released checkpoint. I'm showing its structure rather than its text, because SROIE's redistribution licence is unclear.",
      "The receipt has four fields. Each dot marks the score of that field's worst word. The red line is the token-marginal threshold. Every field's worst word lies beyond it, so all four fields are missed.",
      "The teal ticks are the field-level thresholds that CaliDoc calibrates. All four fields fall inside them.",
      "Nothing about the extractor changed. What changed is the object we calibrate: the complete field, with the document as the exchangeable unit. That is the core idea of CaliDoc."
    ],
    emphasize: "Worst word against threshold. One dot per field decides whether that field is fully covered.",
    why: "Makes the conceptual shift concrete on genuine model output before any equation appears.",
    transition: "Here is how CaliDoc is organized end to end.",
    avoid: "This is one receipt, shown for illustration. The aggregate evidence is Table 2. The marginal threshold is the Table 1 value (mean over three calibration splits); the field thresholds are the released first-split values (Supp. Table S13).",
    sources: "assets/examples/sroie/sroie_test_X51005568889/prediction.json (released checkpoint MHamdan/calidoc-layoutlmv3-large-sroie); main Table 1 (q = 0.0056); Supp. Table S13 (λ_t: ADDRESS 0.1013, COMPANY 0.1402, DATE 0.0224, TOTAL 0.0765)."
  },
  {
    id: 5, title: "CaliDoc framework", time: 45,
    say: [
      "CaliDoc runs entirely on top of a fixed extractor. We take LayoutLMv3-Large with its ordinary argmax decoding, fit a temperature on a held-out early-stopping split, and score every field span by its worst word.",
      "From there, two paths share one calibrated threshold.",
      "The calibration path runs conformal risk control over documents to choose thresholds. For each extracted field it returns a calibrated candidate set.",
      "The audit path reuses the same threshold as a reference, with no new free parameter, and asks where inside a field the escapes land. Is it at the boundaries, or anywhere? It answers that against an explicit null hypothesis.",
      "There's no retraining, no constrained decoding, and no sequence search."
    ],
    emphasize: "One threshold, two uses. Nothing upstream is retrained.",
    why: "Gives the audience a map for the rest of the talk: method, then guarantee, then audit.",
    transition: "Let's start with the calibration path and with exactly what it guarantees.",
    avoid: "Don't imply the audit runs per document. It is a corpus-level test of a field type.",
    sources: "Main Sec. 3 (pp.2–4); temperature on early-stopping split: Sec. 3.1; Supp. S1."
  },
  {
    id: 6, title: "Span score and document-level CRC", time: 75,
    say: [
      "Two design decisions.",
      "First, the score. A span is fully covered at a threshold lambda exactly when its worst word is below lambda. So we score a span by its maximum word score, and full-span coverage becomes a one-dimensional problem. That choice is forced: a mean or a quantile would certify a different event. The price is conservatism on long spans.",
      "Second, the unit. Spans on one page aren't exchangeable, because one misaligned OCR line corrupts several of them. So the document is the exchangeable unit, and a document's loss is the fraction of its gold fields that are missed. That loss is monotone and bounded, so conformal risk control applies: take the smallest lambda whose corrected calibration risk is at most alpha.",
      "The theorem: on a fresh exchangeable document with at least one gold field, the expected fraction of missed gold fields is at most alpha.",
      "[slow] That's an expectation. It doesn't promise that any single document misses at most ten percent, or that each field type is covered at ninety percent. The full-span coverage numbers I'll show next are empirical, not the theorem."
    ],
    emphasize: "\"Expected fraction of missed fields per document.\" Name the quantity explicitly, then list what it is not.",
    why: "A conformal-prediction expert will judge the paper on whether the guaranteed quantity, the exchangeable unit, and the assumptions are stated precisely.",
    transition: "That was the pooled threshold. Field types differ a lot, which brings us to field-conditional thresholds.",
    avoid: "Never say \"guarantees 90% field coverage\". The assumptions are exchangeability of calibration and test documents, and calibration documents disjoint from training, temperature fitting, and model selection.",
    sources: "Main Sec. 3.3, Eqs. 1–3 and the guarantee statement (p.3); Sec. 5.2 (p.8); Supp. S4 (p.3)."
  },
  {
    id: 7, title: "Field-conditional thresholds with pooled fallback", time: 45,
    say: [
      "Running the same risk control per field type gives each type its own threshold. A label enters a word's prediction set when its score is under its own type's threshold, so the rule we deploy and the event we calibrate coincide.",
      "The catch is support. Types with fewer than twenty-five calibration spans fall back to the pooled threshold; otherwise their threshold degenerates and the set becomes the whole label space. On SROIE, all four types are supported. On CORD, most are not.",
      "So, precisely: the pooled bound holds, and each supported type has its own bound. Fallback types inherit only the pooled bound, and aggregate coverage under this mixed rule is empirical."
    ],
    emphasize: "\"Fallback types inherit only the pooled bound.\"",
    why: "Prevents the most likely over-reading, which is a uniform per-field guarantee.",
    transition: "With that precision in place, here are the results across all five settings.",
    avoid: "Don't call the mixed rule \"field-conditional coverage for every field\". The m0 sweep is descriptive, not tuning.",
    sources: "Main Sec. 3.3 'Field-conditional thresholds' (p.3); Supp. S4 (p.3), Table S14 (p.8): 9 supported of 24 listed CORD types; 19.3% of evaluation spans in fallback types (Supp. p.5). Supp. S4.3 prose says '16 of 25'; see Claim Audit discrepancy D1."
  },
  {
    id: 8, title: "Main results (Table 2, Figure 1a)", time: 65,
    say: [
      "Here is the whole picture at the ninety percent target. The bars show empirical full-span coverage.",
      "On SROIE, the token-marginal threshold leaves a realized field-miss risk of 1.000, which means every gold field is missed on every document. Field-conditional SpanCRC brings that to 0.091, against the 0.10 bound on its expectation. Empirically, full-span coverage goes from zero to 0.909 and span CovGap falls from 90 points to 2.1, for sets about ten percent larger.",
      "Repaired SROIE, which I'll come back to, behaves the same way.",
      "On WildReceipt and FUNSD, the marginal threshold was already working, and SpanCRC leaves them roughly where they were. WildReceipt is close to free; FUNSD pays 23 percent in set size for a modest CovGap gain. We don't claim universal improvement: full-span coverage on those two datasets moves down slightly, by half a point or so.",
      "The method helps in proportion to how badly the marginal threshold fails. Then there's CORD."
    ],
    emphasize: "SROIE: risk 1.000 → 0.091, coverage 0.000 → 0.909, CovGap 90.0 → 2.1 pp, about +10% set size.",
    why: "This is the main quantitative evidence. It shows a large repair where marginal calibration fails and nothing breaks where it works.",
    transition: "CORD is our weakest case, so let's look at it directly.",
    avoid: "Only the risk columns relate to the theorem; coverage, CovGap and set size are empirical. Don't read the table row by row. Set-size changes are from Supp. Fig. S1.",
    sources: "Main Table 2 and Fig. 1(a) (p.6), Sec. 4.5 (p.5); Supp. Fig. S1 (set-size %), Table S3 (SROIE 0.90 → 0.99)."
  },
  {
    id: 9, title: "CORD trade-off", time: 55,
    say: [
      "CORD is where we pay. Most field types fall back, so span CovGap improves only from 42.3 to 33.3 points, while average set size more than triples, from 0.90 to 2.92. Thinly supported types get very large thresholds, so their labels enter most sets. Realized risk on its single split is 0.108, just above alpha. That's consistent with a bound on an expectation, but you should see it.",
      "There is a repair that keeps a guarantee. Fix the threshold shape on one half of the calibration documents and choose one level by CRC on the other half. Set size drops to 1.12 and CovGap to 18.4 points. But that keeps only the pooled document-level bound, not a field-conditional one, and it needs enough calibration documents; on FUNSD it inflates sets."
    ],
    emphasize: "Show the cost plainly: 0.90 → 2.92. Then give the fix with its narrower guarantee.",
    why: "Disclosing the weakest result before anyone asks builds credibility with expert reviewers.",
    transition: "Now the second use of the same calibrated threshold: auditing annotations.",
    avoid: "Never present level-calibrated interpolation as a field-conditional bound (Supp. Prop. 3 is pooled only).",
    sources: "Main Sec. 4.5 (pp.5), Sec. 5.2 (p.8), Conclusion limitation 2 (p.8); Supp. S4.3, Proposition 3 and Table S12 (pp.3–5), S8.1 (p.10), Table S14."
  },
  {
    id: 10, title: "Boundary-escape test", time: 70,
    say: [
      "Take the pooled threshold as a reference. A word escapes if its gold tag scores beyond it. In spans where some words escape but not all, we ask whether escapes concentrate on the two boundary words. That is the signature of alignment and BIO errors, where a span starts or ends one word off.",
      "The null is a conditional randomization test. Given each span's length and number of escapes, escaped positions are uniform within the span and independent across spans. Each span's edge count is then hypergeometric, and we get a Monte-Carlo p-value from 5,000 draws. A field that is simply hard escapes everywhere and shows no enrichment.",
      "Two caveats. Cross-span independence is an assumption, since spans on one document can share a defect. So every p-value is paired with a document-blocked bootstrap interval, and we only claim a finding when the two agree. Holm handles multiplicity across fields, but it can't repair that assumption. And spans of two words or fewer carry no positional information."
    ],
    emphasize: "\"Where inside the span\", conditional on how many words escaped. Then name both caveats.",
    why: "The audit is what separates CaliDoc from a pure calibration paper, and its validity conditions must be explicit.",
    transition: "Here is what the test finds, and how we checked it.",
    avoid: "Don't call this a general label-error detector, and don't claim finite-sample validity under arbitrary within-document dependence. Ineligible fields are untested, not clean.",
    sources: "Main Sec. 3.4, Eq. 4 (pp.3–4); Supp. S5 (pp.5–10), Propositions 1–2."
  },
  {
    id: 11, title: "Detected defects and repair validation", time: 60,
    say: [
      "At familywise level 0.10, the test rejects two of SROIE's four eligible fields. ADDRESS shows edge enrichment of 4.07 and COMPANY 1.80, both with a Holm-adjusted p of 0.0008. Their document-blocked intervals exclude one. On the clean benchmarks, none of sixteen eligible field types is rejected, and Holm, BH and BY agree on every split. CORD's SERVICE_PRICE reaches 2.0, but on two informative spans, so the null is wide and nothing is rejected.",
      "Over escaped ADDRESS words, the most frequent prediction is B-ADDRESS at span edges: a one-word boundary offset.",
      "Then the falsification test. SROIE's official Task-3 release ships the gold field strings, so we re-derived the labels independently and evaluated them as a separate lineage. ADDRESS enrichment falls to 0.79, with p = 0.87, and COMPANY to 1.15. Nothing is rejected anywhere.",
      "Calibration didn't repair the data. An independent re-alignment did, and the audit registered it."
    ],
    emphasize: "ADDRESS 4.07 → 0.79 after independent repair. Say clearly that the repair is not CaliDoc.",
    why: "Positive detection plus a negative after repair is the strongest evidence the audit measures something real.",
    transition: "How does this relate to the closest prior work?",
    avoid: "The repaired results come from a separately evaluated annotation lineage; the original gate still failed (94.45% < 95%). The positive real-data validation is one dataset. ADDRESS and COMPANY share the same receipts.",
    sources: "Main Sec. 4.6 and Fig. 1(b) (p.6), Sec. 4.9 (p.7); Supp. Table S17 (p.11), Table S18 (p.12), S6 (p.10)."
  },
  {
    id: 12, title: "Closest prior work", time: 50,
    say: [
      "The closest prior work is Rombach and Mehdiyev, published in IJDAR in 2026. They brought split conformal prediction to key information extraction, scoring each entity by its first token, with marginal coverage over entity instances. So conformal KIE isn't new, and we don't claim it is.",
      "What we change is the guaranteed object: the complete field, as a document-level risk. Mondrian and clustered CP fix class imbalance, but they still certify tokens.",
      "At their operating point on WildReceipt, our probabilities give similar coverage with smaller sets. We state that comparison rather than rank it, because backbone, score, schema and splits differ. The more telling number: on SROIE, the first-token unit reaches 0.897 while full-span coverage is 0.856."
    ],
    emphasize: "\"Same tool, different guaranteed object.\" Name the prior work first, and generously.",
    why: "Reviewers will ask about novelty. The answer is the unit of the guarantee plus the audit, not \"first CP for KIE\".",
    transition: "Here are the limits of what we've shown.",
    avoid: "No \"first\", no SOTA, no head-to-head ranking. The proxy-to-field gap depends on the dataset and the operating point.",
    sources: "Main Sec. 2.2 (p.2), Sec. 4.4.1 (pp.4–5); Supp. Table S9 (p.6)."
  },
  {
    id: 13, title: "Limitations", time: 45,
    say: [
      "Six limitations, all of them in the paper.",
      "First, the guarantee is a risk, not a coverage, and a single split can exceed alpha, as CORD's does.",
      "Second, field-conditional calibration is limited by support, and CORD shows the cost.",
      "Third, the audit is blind on spans of two words or fewer, which leaves short-span schemas like WildReceipt largely untested.",
      "Fourth, the null assumes independence across spans.",
      "Fifth, the real-data positive rests on one corpus, SROIE, and both flagged fields share the same receipts.",
      "Sixth, the scope is four English receipt and form corpora and one backbone family, at two scales. Multilingual, long-document, and OCR-free settings are untested."
    ],
    emphasize: "Deliver this calmly and completely. It is a strength slide, not an apology.",
    why: "Pre-empts the hardest questions and shows that the claims are scoped.",
    transition: "So what should you take away?",
    avoid: "Don't add limitations that aren't in the paper, and don't soften the ones that are.",
    sources: "Main Sec. 6 'Limitations' (p.8), Sec. 3.4 (p.4), Sec. 4.6 (p.6); Supp. S2.1 (Base backbone)."
  },
  {
    id: 14, title: "Takeaways and resources", time: 45,
    say: [
      "Three things to take away.",
      "Token-level reliability is not complete-field reliability. On SROIE, 89.8% of tokens were covered and zero of 420 fields were.",
      "Make the delivered field the unit. Conformal risk control over documents bounds the expected fraction of missed fields, without retraining the extractor.",
      "And use the same calibration to audit the annotations. A positional test with an explicit null localized a real annotation defect, and an independent repair silenced it.",
      "Code, released checkpoints, the supplement, and a Colab demo are linked here. This work was funded by an NSERC Discovery Grant.",
      "[close] Report reliability for the object you actually deliver. Thank you, and I'm happy to take questions."
    ],
    emphasize: "Closing line: \"Report reliability for the object you actually deliver.\"",
    why: "A memorable line tied directly to the contribution.",
    transition: "Q&A. Backup slides B1–B10 follow.",
    avoid: "Don't introduce new numbers here.",
    sources: "Main Sec. 6 (p.8), Acknowledgment (p.8); Supp. Table S40 (p.21)."
  },
  // ---------------- Backup ----------------
  {
    id: 15, title: "Backup divider", time: 0, backup: true,
    say: ["Backup slides, used only in Q&A."],
    emphasize: "", why: "", transition: "", avoid: "", sources: ""
  },
  {
    id: 16, title: "B1 · Mondrian on SROIE", time: 0, backup: true,
    say: [
      "This answers \"why not just use class-conditional CP?\" On SROIE, Mondrian calibration repairs per-class token coverage cheaply. Every class returns to between 87 and 100 percent, at average set size 0.90, and CovGap falls from 71.1 to 3.4 points.",
      "Two things to note. First, it still certifies tokens, not fields. Second, the price is a quantile spread of a factor of 367, from 0.0027 for O to 0.9996 for I-TOTAL. That spread is a screening signal, not evidence, because part of it is thin support. That's why the audit uses a statistic with a null."
    ],
    emphasize: "Mondrian fixes per-class token coverage; it is still a token guarantee.",
    why: "Distinguishes CaliDoc from class-conditional CP.",
    transition: "", avoid: "Don't present the quantile spread as a label-noise test.",
    sources: "Main Table 1 and Sec. 4.4.2 (p.5)."
  },
  {
    id: 17, title: "B2 · Why the max score", time: 0, backup: true,
    say: [
      "If you calibrate the pooled CRC threshold on other span scores, each lands near nominal on its own event but not on the delivered one. On SROIE, the mean score achieves 0.898 on its own event, but true full-span coverage is only 0.762 and realized document risk is 0.238, compared with 0.114 under the max.",
      "The gap grows with span length; length and span score correlate at Spearman 0.59 on SROIE against 0.04 on WildReceipt, where spans are at most two words long and all scores coincide."
    ],
    emphasize: "Other scores certify other events; only the max certifies complete-field coverage.",
    why: "Answers \"why max and not mean?\"", transition: "", avoid: "These are diagnostic ablations, not proposed alternatives.",
    sources: "Main Sec. 3.3 (p.3), Sec. 4.5 (p.5); Supp. S4.1–S4.2, Tables S10–S11 (p.6)."
  },
  {
    id: 18, title: "B3 · Small-calibration controls", time: 0, backup: true,
    say: [
      "Could the per-class collapse just be a small-sample artifact? We ran three controls.",
      "First, a feasibility bound: a class-conditional quantile needs at least 8 calibration tokens at alpha 0.10, and classes below that are marked rather than scored.",
      "Second, a matched-n control on clean CORD classes: coverage is 0.810, 0.910, 0.954, 0.922, and 0.905 at n equal to 5, 10, 20, 50, and 100. That's a nine-point cost at n = 5, nowhere near a collapse to zero.",
      "Third, stronger baselines on CORD: clustered CP degenerates toward marginal, 45.9 versus 45.8 points; interpolation gets 32.2; full class-conditional gets 16.4 but grows set size from 0.90 to 2.49."
    ],
    emphasize: "Small n costs about nine points, not one hundred.",
    why: "Answers the calibration-size objection.", transition: "", avoid: "",
    sources: "Main Sec. 4.7 (p.7); Supp. Tables S28–S29."
  },
  {
    id: 19, title: "B4 · Audit size and power", time: 0, backup: true,
    say: [
      "Size: when we redraw escape positions uniformly over 200 replicates, the familywise rejection rate under Holm is 0.025 on FUNSD and zero on WildReceipt and CORD, against 0.140 uncorrected on FUNSD.",
      "Power, under end-extension corruption: a single corrupted FUNSD QUESTION span is detected at rate 0.53 over 30 seeds, two spans at 0.73, and twenty at 0.93. WildReceipt never exceeds zero, which is the short-span blind spot. Start-side shifts reach at most 0.20.",
      "One disclosure: FUNSD QUESTION has a borderline signal on clean labels, raw p 0.036 and Holm 0.109, which is not a finding.",
      "Adding a pre-specified interior mask and correcting over all 36 field-mask pairs still rejects exactly the two SROIE fields, and only under the edge mask."
    ],
    emphasize: "Holm is doing visible work; the blind spot is structural.", why: "Answers \"how do you know the test is calibrated and has power?\"",
    transition: "", avoid: "Synthetic size and power are on our own probabilities. They are not a benchmark against other detectors.",
    sources: "Main Sec. 4.6 'Size and power' (pp.6–7); Supp. Tables S16, S21–S23."
  },
  {
    id: 20, title: "B5 · Label-error detectors", time: 0, backup: true,
    say: [
      "Under planted CORD noise, our coverage-deficit diagnostic is weaker than dedicated detectors at token-level ranking, and we say so directly. AUPRC is 0.007 to 0.169 across six noise conditions, against 0.047 to 0.951 for Cleanlab and 0.117 to 0.859 for AUM. Scoring a class-level statistic per token is the wrong use of it.",
      "At class level, under class-targeted noise, it ranks the corrupted class 6th to 12th of 54, against 21st to 46th for the baselines, but never in the top 3, and under uniform or BIO-shift noise it ranks 48th and 38th. Localization comes from the boundary-escape test, which conditions on position and carries a null."
    ],
    emphasize: "Different job: a field-level hypothesis test, not a token ranker.",
    why: "Answers \"do you beat Cleanlab?\" No, and that isn't the claim.", transition: "", avoid: "Don't claim superiority over label-error detectors.",
    sources: "Main Sec. 4.8 (p.7); Supp. S7, Table S26 (p.17)."
  },
  {
    id: 21, title: "B6 · Generality", time: 0, backup: true,
    say: [
      "LayoutLMv3-Base under the identical protocol reproduces the SROIE collapse on all three seeds. Full-span coverage is 0.000, repaired to 0.901, and risk is 1.000, repaired to 0.099. The audit rejects exactly ADDRESS and COMPANY on all three seeds, and the clean-benchmark pattern matches. The one difference is that Base pays more set size on WildReceipt, 1.71 against 0.93.",
      "With a clean calibration protocol, retraining WildReceipt with calibration carved from the training partition and the official test set used only for final evaluation, F1 is 94.05 against 94.38 and every conformal conclusion replicates."
    ],
    emphasize: "Two scales, one family. Broader architectures are untested.", why: "Answers backbone and protocol questions.", transition: "",
    avoid: "Don't claim architecture-general results; OCR-free models are untested.",
    sources: "Main Sec. 4.5 last paragraph (p.5); Supp. S2.1, Table S3 (pp.2–4)."
  },
  {
    id: 22, title: "B7 · SROIE alignment and repair lineage", time: 0, backup: true,
    say: [
      "The pre-registered gate required at least 95% exact contiguous string-to-token alignment per split. It failed at 94.68% on train and 94.02% on test, with a pooled rate of 94.45%. The result was identical across three independent data lineages, including the official competition site, and that is why SROIE is a case study rather than a benchmark.",
      "Punctuation-insensitive matching recovers another 2.80% and numeric normalization another 0.13%, reaching 97.38%. Address fails most often: 85.6% exact on test, against 95.1 to 99.1% for the other fields. We rebuilt under the relaxed matchers as a separate lineage; the gate decision stands.",
      "At token level, the repair looks mixed: B-ADDRESS rises from 0 to 0.444 and I-ADDRESS falls from 0.959 to 0.261. At the span and position level it is a clean confirmation."
    ],
    emphasize: "The repair is a separate lineage; it does not retroactively pass the gate.", why: "Answers \"how strong is the repair evidence?\"", transition: "",
    avoid: "Don't say the repaired data passes the gate.",
    sources: "Main Sec. 4.9 (p.7); Supp. S6, Tables S24–S25 (pp.10, 16)."
  },
  {
    id: 23, title: "B8 · Reading α in deployment", time: 0, backup: true,
    say: [
      "Alpha equal to 0.10 bounds the expected per-document fraction of missed gold fields. On SROIE, with four required fields per receipt, that means at most 0.4 expected missed fields per receipt, and no individual receipt is guaranteed.",
      "Choosing alpha is a costing exercise: the expected review burden is roughly alpha times fields per document times volume, weighed against the cost of a miss.",
      "Set shape matters. On CORD, 100% of tokens carry multi-label sets, so every field arrives with alternatives and goes to triage. SROIE and WildReceipt stay near-singleton, at 99.0% and 92.9%, and there the cost is empty sets, 6.3% of WildReceipt tokens, which are forced abstentions."
    ],
    emphasize: "A bound on an average, not on a receipt.", why: "Answers deployment questions.", transition: "",
    avoid: "Per-field or adaptive alpha would need a guarantee we do not have.",
    sources: "Main Sec. 5.2 (p.8); Supp. S8.1 (p.10)."
  },
  {
    id: 24, title: "B9 · CaliDoc in action (CORD)", time: 0, backup: true,
    say: [
      "This is the released CORD checkpoint on a CORD v2 test receipt, licensed CC BY 4.0. The extractor returns 25 fields and 24 match the gold labels. Because CORD sets are multi-label, all 25 fields are routed to review.",
      "The one incorrect extraction, Coupon 100,000, is predicted as SUB_TOTAL.ETC where the gold label is TOTAL.TOTAL_ETC. Its worst word scores 0.80 against the pooled threshold 0.189, so its calibrated set excludes the prediction. It is the only field in the conflict state.",
      "This is one illustrative case, not a sensitivity or specificity result, and CaliDoc does not guarantee that incorrect fields are detected."
    ],
    emphasize: "Routing is operational, not a correctness probability.", why: "Shows the released software on a licence-cleared document.", transition: "",
    avoid: "Don't present routing states as additional guarantees, and don't use this one receipt as an accuracy claim.",
    sources: "Supp. S12 (pp.20–21); assets/examples/cord/cord_test_0014 (CC BY 4.0)."
  },
  {
    id: 25, title: "B10 · Calibration and robustness", time: 0, backup: true,
    say: [
      "After temperature scaling, ECE over three seeds is 1.27% on CORD, 1.81% on WildReceipt, and 6.37% on FUNSD.",
      "Under nine image perturbations at three severities, with thresholds held at their clean values, mean F1 drop is 0.56 points on FUNSD and mean coverage drop is below 0.15 points everywhere, with a worst cell of 1.53. This coverage is empirical, because perturbation breaks exchangeability.",
      "MC Dropout hurts: CORD ECE goes from 1.57 to 2.77% and coverage from 87.9 to 78.6%, at twenty times the inference cost.",
      "On CORD, 9.9% of tokens get an empty LAC set, and 68.2% of those would have been classified correctly."
    ],
    emphasize: "Coverage under shift is empirical, not guaranteed.", why: "Answers distribution-shift and baseline questions.", transition: "",
    avoid: "No guarantee under distribution shift.",
    sources: "Main Sec. 4.4 (p.4), Sec. 4.10 (p.7); Supp. S8–S9."
  }
];
