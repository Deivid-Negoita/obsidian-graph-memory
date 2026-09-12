# obsidian-graph-memory

Give an LLM a memory of your notes that costs **~400 tokens and ~2 ms**, instead
of a search session that grows with the vault.

It reads a folder of Obsidian markdown, walks the links you already wrote into a
SQLite graph, and injects the relevant facts into the prompt *before* the model
is invoked — as a [Claude Code](https://claude.com/claude-code)
`UserPromptSubmit` hook, or as a CLI you can pipe anywhere.

Stdlib Python only. No dependencies, no vector database, no embedding model, and
no model call anywhere in the retrieval path.

```
$ python graph_memory.py --vault example-vault recall "how is a wiki different from RAG"

memory: 8 facts recalled in 3 ms

Compounding Knowledge --[related_to]--> LLM Wiki Pattern            (concepts/Compounding Knowledge.md)
LLM Wiki Pattern --[related_to]--> Session Cache                    (concepts/LLM Wiki Pattern.md)
LLM Wiki Pattern --[references]--> Retrieval Augmented Generation   (concepts/LLM Wiki Pattern.md)
LLM Wiki Pattern --[cites]--> Karpathy Thread                       (concepts/LLM Wiki Pattern.md)
Graph Recall --[related_to]--> Retrieval Augmented Generation       (concepts/Graph Recall.md)

where:
  Graph Recall: Walk edges instead of ranking chunks. Cheap, deterministic, and explainable
  LLM Wiki Pattern: A durable, linked knowledge base an LLM reads before answering
  Session Cache: Everything the model knows inside one conversation and forgets at the end of it
```

## Try it in 30 seconds

```bash
git clone https://github.com/Deivid-Negoita/obsidian-graph-memory
cd obsidian-graph-memory

python graph_memory.py selftest                     # builds a fixture, walks it, asserts
python graph_memory.py --vault example-vault build  # 8 notes -> 9 entities, 16 relations
python graph_memory.py --vault example-vault recall "why does knowledge compound"
```

Then point it at your own vault:

```bash
python graph_memory.py --vault ~/my-vault build
python graph_memory.py --vault ~/my-vault status
```

## Wire it into Claude Code

Add to `.claude/settings.json`. The hook gets the prompt on stdin and prints
context on stdout, which Claude Code prepends to the turn:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/graph_memory.py hook"
          }
        ]
      }
    ]
  }
}
```

The hook takes the vault path from the payload's `cwd`, so it works per-project.
Set `GRAPH_MEMORY_VAULT` to override. Rebuild after adding notes — `build` on a
2,400-note vault takes about a minute, and `status` tells you when the graph is
behind the notes.

**It can never fail your turn.** Every path in the hook is wrapped and exits 0.
A hook that crashes the prompt is worse than a hook that occasionally recalls
nothing.

## Why a graph and not embeddings

Similarity is not a relationship. RAG is strong on a large unstructured corpus
and weak on exactly the question a knowledge base exists to answer — *how does
this connect to that* — because the nearest chunks to "X" are other chunks about
X, not the thing X was linked to.

An Obsidian vault has already solved the hard half: **a human wrote the edges.**
Every entity and every relation here is read off frontmatter or a wikilink that
someone wrote on purpose:

| written in the note | becomes |
| --- | --- |
| `related:` | `related_to` |
| `sources:` | `cites` |
| `[[wikilink]]` | `references` |
| `domain:` | `in_domain` |
| folder's `_index.md` | `part_of` |

Nothing is inferred, so nothing has to be re-checked. Every injected fact names
the note it came from, and a dead link resolves to nothing rather than to an
invented node.

The walk is three tables and one recursive CTE. It runs as code before the model
is invoked, so the model spends its turn answering rather than retrieving.

## The parts that had to be added

The schema, the recursive `WALK` query and the whole-name seeding rule are from
[graph-memory-starter](https://github.com/Glitch-Cat-Club/graph-memory-starter)
(MIT), which models a small hand-written corpus. Pointing it at a real
2,400-note vault surfaced three problems. Numbers below are measured on that
vault, not estimated.

**Walk order is mostly plumbing.** Over 12 real questions, 77 facts recalled and
42 of them (54%) were a hub or a grouping edge. `references` alone was 48% of
4,385 relations, nearly all of it an index or a log linking a note — so a 3-hop
walk from any seed reached the entire vault. Fixed by *scoring* triples rather
than truncating in walk order: predicate weight, a hub penalty above degree 60
(median degree is 6, p95 is 46), and decay with hop distance from the seed.
Nothing is dropped — a hub edge is the only fact some questions have, and
demoting beats deleting when the alternative is an empty recall.

**A journal is chronology, not topology.** A wikilink in an append-only log means
"this was touched on this date", never "these two things are related". Degree
alone does not catch it: rotating a log file drops the live log under the hub
threshold and its plumbing promptly comes back — measured 12% → 18% of recalled
facts. Journal-stemmed sources are penalised by name.

**Exact seeding is correct and usually empty.** "Why does knowledge compound"
names no entity in full, so upstream's whole-name match finds no seed and recall
returns nothing while the note sits in the graph. Word seeding closes it: an
entity whose name shares a significant word with the question becomes an entry
point, prefix-matched so "compound" reaches "Compounding Knowledge" without
shipping a stemmer. Ranking then decides whether it deserved to be — and an
unrelated question ("what time is dinner") still recalls nothing.

One more, cheaper to state than to rediscover: **entity names must mean exactly
one thing.** Edges and seeds resolve by normalised name, so a second entity
claiming a name already held by a different type is qualified with that type
rather than silently inheriting the first one's facts.

## Commands

| command | does |
| --- | --- |
| `build` | notes → `.graph-memory/graph.db` (`--scope` to index a subfolder) |
| `recall "..."` | what the hook would inject (`--hops`, `--top-k`) |
| `hook` | `UserPromptSubmit` hook, JSON on stdin |
| `status` | counts, staleness, busiest nodes |
| `selftest` | builds a fixture and asserts the things that actually break |

## Credit

Schema, `WALK` query and seeding rule: **Glitch-Cat-Club/graph-memory-starter**
(MIT). Extraction, ranking, word seeding and the hook: this repo.

Extracted from a larger personal Obsidian vault where it runs on every prompt.

MIT — Deivid Bogdan Negoita.
