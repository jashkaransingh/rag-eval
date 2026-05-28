# RAG evaluation report

- adapter: `ragqa`
- judge: `stub`
- cases: 6
- latency p50: 0.9ms, p95: 0.9ms

## metrics

| metric | mean | std |
|--------|------|-----|
| hit@3 | 1.000 | 0.000 |
| recall@3 | 1.000 | 0.000 |
| precision@3 | 0.333 | 0.000 |
| mrr | 0.889 | 0.248 |
| ndcg@3 | 0.917 | 0.186 |
| faithfulness | 0.630 | 0.088 |
| answer_relevance | 0.526 | 0.178 |
| context_precision | 0.742 | 0.094 |
| answer_correctness | 0.338 | 0.058 |

## per-case results

### case 1
> What was Acme's Q3 revenue and what drove the growth?

**answer:** Based on the document: # Acme Robotics Quarterly Update [1]

scores: hit@3=1.000, recall@3=1.000, precision@3=0.333, mrr=1.000, ndcg@3=1.000, faithfulness=0.650, answer_relevance=0.440, context_precision=0.720, answer_correctness=0.300
relevant ids: ['acme_q3.md:0']
retrieved ids: ['acme_q3.md:0', 'mark_vii_specs.md:2', 'acme_q3.md:2']

---

### case 2
> What software platform was released in September and what changed?

**answer:** Based on the document: Our software platform, AcmeOS 4.2, was released in September. The major change [1]

scores: hit@3=1.000, recall@3=1.000, precision@3=0.333, mrr=1.000, ndcg@3=1.000, faithfulness=0.680, answer_relevance=0.767, context_precision=0.767, answer_correctness=0.378
relevant ids: ['acme_q3.md:1']
retrieved ids: ['acme_q3.md:1', 'mark_vii_specs.md:1', 'acme_q3.md:2']

---

### case 3
> Why did the Mark VII shipment slip?

**answer:** Based on the document: The Mark VII shipment timeline slipped two weeks due to a firmware issue with [1]

scores: hit@3=1.000, recall@3=1.000, precision@3=0.333, mrr=1.000, ndcg@3=1.000, faithfulness=0.700, answer_relevance=0.767, context_precision=0.767, answer_correctness=0.450
relevant ids: ['acme_q3.md:2']
retrieved ids: ['acme_q3.md:2', 'mark_vii_specs.md:0', 'acme_q3.md:1']

---

### case 4
> How heavy is the Mark VII and what is its payload capacity?

**answer:** Based on the document: # Mark VII Technical Specifications [1]

scores: hit@3=1.000, recall@3=1.000, precision@3=0.333, mrr=1.000, ndcg@3=1.000, faithfulness=0.564, answer_relevance=0.440, context_precision=0.720, answer_correctness=0.300
relevant ids: ['mark_vii_specs.md:0']
retrieved ids: ['mark_vii_specs.md:0', 'acme_q3.md:0', 'acme_q3.md:2']

---

### case 5
> What is the battery life of the Mark VII under peak load?

**answer:** Based on the document: ## Power [1]

scores: hit@3=1.000, recall@3=1.000, precision@3=0.333, mrr=1.000, ndcg@3=1.000, faithfulness=0.467, answer_relevance=0.300, context_precision=0.900, answer_correctness=0.300
relevant ids: ['mark_vii_specs.md:1']
retrieved ids: ['mark_vii_specs.md:1', 'mark_vii_specs.md:0', 'acme_q3.md:2']

---

### case 6
> Which ROS versions does AcmeOS 4.2 support?

**answer:** Based on the document: Our software platform, AcmeOS 4.2, was released in September. The major change [1]

scores: hit@3=1.000, recall@3=1.000, precision@3=0.333, mrr=0.333, ndcg@3=0.500, faithfulness=0.720, answer_relevance=0.440, context_precision=0.580, answer_correctness=0.300
relevant ids: ['mark_vii_specs.md:2']
retrieved ids: ['acme_q3.md:1', 'mark_vii_specs.md:0', 'mark_vii_specs.md:2']

---
