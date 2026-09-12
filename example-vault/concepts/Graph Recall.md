---
type: concept
domain: knowledge-systems
related:
  - "[[Retrieval Augmented Generation]]"
---
Walk edges instead of ranking chunks. Cheap, deterministic, and explainable:
every fact injected can be traced to the note and the link it came from.
Runs as SQL before the model is invoked, so it costs a fixed token budget.
