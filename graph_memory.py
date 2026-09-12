#!/usr/bin/env python
"""Knowledge-graph memory over a folder of Obsidian notes: three SQLite tables,
one recursive query, one prompt hook.

Why it exists
-------------
Answering "what does X connect to" by search costs a grep, several reads, and a
multi-hop chain the model has to hold in its head. The same answer walked as SQL
costs ~2 ms and a fixed ~400-token injection that does not grow with the vault.
The traversal runs as code *before* the model is invoked, so the model spends
its turn answering rather than retrieving.

Where the edges come from
-------------------------
Nothing here is inferred. Every entity and every edge is read off frontmatter or
a wikilink that a human already wrote:

    related:  -> related_to      sources: -> cites
    [[link]]  -> references      domain:  -> in_domain
    folder    -> part_of  (the folder's _index.md)

That matters more than it sounds. A graph an LLM guessed at is a graph you have
to re-check; a graph walked off links someone wrote on purpose is evidence.

Usage
-----
    python graph_memory.py build   [--vault PATH]   # notes -> .graph-memory/graph.db
    python graph_memory.py recall "..."             # what the hook would inject
    python graph_memory.py status                   # counts, staleness
    python graph_memory.py hook                     # UserPromptSubmit hook (stdin)
    python graph_memory.py selftest                 # build + walk a fixture

Credit
------
The schema, the recursive WALK query and the seeding rule are
Glitch-Cat-Club/graph-memory-starter (MIT). This file is what happened when that
starter was pointed at a real 2,400-note vault instead of a hand-written corpus:
the extraction, the ranking, and the hook are the parts that had to be added
before recall was useful rather than merely correct. Comments marked "measured"
are numbers from that vault, not estimates.

Stdlib only. No dependencies, no vector database, no model call anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

STATE_DIRNAME = ".graph-memory"
DB_NAME = "graph.db"

# Closed vocabulary, fixed before anything was written -- the starter's first
# discipline. Entity types are the notes' own `type:` values, uppercased, plus
# the grouping nodes that make multi-hop questions answerable.
NOTE_TYPES = {
    "concept", "entity", "source", "tool", "reference", "meta",
    "session", "comparison", "overview", "question", "redirect",
}
PREDICATES = ("related_to", "cites", "references", "in_domain", "part_of")

# Names that would seed on almost every prompt. A seed is a whole-word match on
# the user's words, so a node called "index" drags the whole vault in.
STOP_NAMES = {
    "index", "log", "hot", "overview", "note", "notes", "meta", "source",
    "sources", "concept", "concepts", "entity", "entities", "wiki", "vault",
    "readme", "home", "map", "dashboard", "tools", "tool",
}
# 3, not 4. Tools get named in three letters -- rtk, qmd, gev -- and at 4 none of
# them could ever become a seed, so "what does rtk do" recalled nothing while the
# note about rtk sat in the graph.
MIN_SEED_LEN = 3

SKIP_DIRS = {".git", ".obsidian", ".trash", STATE_DIRNAME, "node_modules",
             "__pycache__", ".venv"}

SCHEMA = """
CREATE TABLE entities  (id TEXT PRIMARY KEY,   -- uuid5(type + normalised name)
                        name TEXT, type TEXT,
                        description TEXT, source_doc TEXT);
CREATE TABLE relations (source_id TEXT, target_id TEXT,
                        predicate TEXT, source_doc TEXT);
CREATE TABLE aliases   (entity_id TEXT, alias TEXT);
CREATE INDEX rel_src ON relations(source_id);
CREATE INDEX rel_tgt ON relations(target_id);
"""

# Upstream's walk, unchanged. Seed entities, expand k hops in either direction,
# then return every relation whose both ends landed inside the walk -- with
# `near`, the hop distance from the closest seed, which the ranking below uses.
WALK = """
WITH RECURSIVE walk(entity_id, depth) AS (
  SELECT id, 0 FROM entities WHERE id IN ({seeds})
  UNION
  SELECT CASE WHEN r.source_id = w.entity_id
              THEN r.target_id ELSE r.source_id END,
         w.depth + 1
  FROM relations r JOIN walk w
    ON w.entity_id IN (r.source_id, r.target_id)
  WHERE w.depth < ?
)
SELECT e1.name, r.predicate, e2.name, r.source_doc,
       MIN((SELECT MIN(depth) FROM walk WHERE entity_id = r.source_id),
           (SELECT MIN(depth) FROM walk WHERE entity_id = r.target_id)) AS near
FROM relations r
JOIN entities e1 ON e1.id = r.source_id
JOIN entities e2 ON e2.id = r.target_id
WHERE r.source_id IN (SELECT entity_id FROM walk)
  AND r.target_id IN (SELECT entity_id FROM walk)
ORDER BY near
"""

DEGREE_SQL = """
SELECT e.name, SUM(d) FROM (
  SELECT source_id AS id, COUNT(*) AS d FROM relations GROUP BY 1
  UNION ALL SELECT target_id, COUNT(*) FROM relations GROUP BY 1
) x JOIN entities e ON e.id = x.id GROUP BY e.name
"""


def normalise(name: str) -> str:
    return name.lower().strip().replace(" ", "_")


def entity_id(type_: str, name: str) -> str:
    # "Ops Manager" in doc 1 and doc 4 -> the same node. No ML, no lookup.
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{type_}:{normalise(name)}"))


# --------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------

WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass
class Note:
    rel: str
    stem: str
    front: dict = field(default_factory=dict)
    body: str = ""
    links: list = field(default_factory=list)


def _scalar(raw: str):
    """One frontmatter value. Inline lists stay lists, everything else is str."""
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        return [p.strip().strip("\"'") for p in text[1:-1].split(",") if p.strip()]
    return text.strip("\"'")


def parse_frontmatter(text: str) -> tuple:
    """Minimal YAML: scalars, inline lists, and block lists. That is the whole
    subset Obsidian frontmatter actually uses for links and types, and it keeps
    this file dependency-free -- a hook that runs on every prompt should not be
    able to fail on someone else's PyYAML version.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block, body = text[3:end], text[end + 4:]
    front, key = {}, None
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- ") and key:
            front.setdefault(key, [])
            if isinstance(front[key], list):
                front[key].append(_scalar(line.lstrip()[2:]))
            continue
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, raw = line.partition(":")
            key = key.strip()
            front[key] = _scalar(raw) if raw.strip() else []
    return front, body


def read_notes(vault: Path, scope: str = "") -> dict:
    notes = {}
    for path in vault.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(vault).as_posix()
        if scope and not rel.startswith(scope):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        front, body = parse_frontmatter(text)
        notes[rel] = Note(rel=rel, stem=path.stem, front=front, body=body,
                          links=WIKILINK.findall(body))
    return notes


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [v for v in value if isinstance(v, str) and v.strip()]


def _clean_link(value: str) -> str:
    """`- "[[Some Page|alias]]"` -> `Some Page`."""
    text = value.strip().strip('"').strip("'")
    m = re.match(r"^\[\[([^\]|#]+)", text)
    text = (m.group(1) if m else text).strip()
    return text.split("|")[0].split("#")[0].strip()


def _h1(note: Note) -> str:
    """The title a frontmatter-less note actually wrote for itself."""
    for raw in note.body.splitlines()[:12]:
        if raw.startswith("# "):
            return raw[2:].strip()
    return ""


def _title(note: Note) -> str:
    """The one name this note is known by. Must agree everywhere, or an edge
    resolves onto a different entity."""
    front_title = str(note.front.get("title") or "").strip()
    return front_title or note.stem


def _description(note: Note) -> str:
    desc = str(note.front.get("description") or "").strip()
    if desc:
        return desc[:200]
    for line in note.body.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", ">", "-", "|", "!", "[")):
            return line[:200]
    return ""


def build_lookup(notes: dict) -> dict:
    """How Obsidian resolves `[[X]]`: by stem, by full path, and by a declared
    alias. A dead link resolves to nothing and is dropped -- never to an
    invented node.

    Aliases are not a nicety here. Obsidian will not resolve `[[LLM Wiki
    Pattern]]` to `llm-wiki-pattern.md` on its own, so a vault that writes
    titles differently from filenames carries `aliases:` to bridge them, and a
    lookup that ignores them silently drops those edges.
    """
    lookup: dict = {}
    for rel, note in notes.items():
        keys = [note.stem.lower(), rel.lower().removesuffix(".md")]
        keys += [_clean_link(a).lower() for a in _as_list(note.front.get("aliases"))]
        title = str(note.front.get("title") or "").strip().lower()
        if title:
            keys.append(title)
        for key in keys:
            if key:
                lookup.setdefault(key, []).append(rel)
    return lookup


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def collect(notes: dict) -> tuple:
    """Notes -> (nodes, edges, aliases)."""
    nodes: dict = {}
    edges: list = []
    aliases: list = []
    lookup = build_lookup(notes)
    taken: dict = {}

    def node(name: str, type_: str, desc: str = "", doc: str = "") -> str:
        """Add an entity and return the name edges should point at.

        Edge endpoints and seeds resolve by normalised name, so a name may only
        mean one thing. A second entity wanting a name already held by a
        different type is qualified with that type instead of silently
        inheriting the first one's facts.
        """
        key = name.lower()
        if taken.get(key) not in (None, type_):
            name = f"{name} ({type_.lower()})"
            key = name.lower()
        taken[key] = type_
        if key not in nodes:
            nodes[key] = {"name": name, "type": type_,
                          "description": desc, "source_doc": doc}
        elif desc and not nodes[key]["description"]:
            nodes[key]["description"] = desc
        return name

    for rel, note in notes.items():
        raw_type = str(note.front.get("type") or "note").lower()
        ntype = raw_type.upper() if raw_type in NOTE_TYPES else "NOTE"
        title = _title(note)
        node(title, ntype, _description(note), rel)

        for alias in _as_list(note.front.get("aliases")):
            aliases.append({"entity": title, "alias": _clean_link(alias)})
        if note.stem.lower() != title.lower():
            aliases.append({"entity": title, "alias": note.stem})

        for domain in _as_list(note.front.get("domain")):
            edges.append({"source": title, "predicate": "in_domain",
                          "target": node(domain, "DOMAIN", "", rel), "doc": rel})

    def target_title(raw: str, source_rel: str):
        key = _clean_link(raw).lower()
        for dest in lookup.get(key, []):
            if dest in notes:
                return _title(notes[dest])
        return None

    # Second pass: edges between notes, resolved the way Obsidian resolves a
    # wikilink, so a dead link is dropped rather than inventing a node.
    for rel, note in notes.items():
        title = _title(note)
        seen = set()

        for key, predicate in (("related", "related_to"), ("sources", "cites")):
            for raw in _as_list(note.front.get(key)):
                dest = target_title(raw, rel)
                if dest and dest != title and (dest, predicate) not in seen:
                    seen.add((dest, predicate))
                    edges.append({"source": title, "predicate": predicate,
                                  "target": dest, "doc": rel})

        for raw in note.links:
            dest = target_title(raw, rel)
            if not dest or dest == title:
                continue
            # An explicit related:/sources: edge is the better-typed one.
            if any((dest, p) in seen for p in ("related_to", "cites", "references")):
                continue
            seen.add((dest, "references"))
            edges.append({"source": title, "predicate": "references",
                          "target": dest, "doc": rel})

        folder = rel.rsplit("/", 1)[0] if "/" in rel else ""
        parent = notes.get(f"{folder}/_index.md") if folder else None
        if parent is not None:
            ptitle = _title(parent)
            if ptitle != title:
                edges.append({"source": title, "predicate": "part_of",
                              "target": ptitle, "doc": rel})

    return list(nodes.values()), edges, aliases


def write_db(db_file: Path, nodes: list, edges: list, aliases: list) -> dict:
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_file.unlink(missing_ok=True)
    with closing(sqlite3.connect(db_file)) as db:
        db.executescript(SCHEMA)
        for n in nodes:
            db.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?,?)",
                       (entity_id(n["type"], n["name"]), n["name"], n["type"],
                        n.get("description", ""), n.get("source_doc", "")))

        by_name = {normalise(name): eid
                   for eid, name in db.execute("SELECT id, name FROM entities")}
        for e in edges:
            s = by_name.get(normalise(e["source"]))
            t = by_name.get(normalise(e["target"]))
            if s and t and s != t:
                db.execute("INSERT INTO relations VALUES (?,?,?,?)",
                           (s, t, e["predicate"], e.get("doc", "")))

        for a in aliases:
            eid = by_name.get(normalise(a["entity"]))
            if eid and a["alias"].strip():
                db.execute("INSERT INTO aliases VALUES (?,?)",
                           (eid, a["alias"].strip()))

        # Seeding is a whole-word match, so anything too short or too generic is
        # dropped here rather than poisoning every recall.
        db.execute(
            "DELETE FROM aliases WHERE LENGTH(alias) < ? OR LOWER(alias) IN (%s)"
            % ",".join("?" * len(STOP_NAMES)),
            (MIN_SEED_LEN, *sorted(STOP_NAMES)),
        )
        db.commit()
        counts = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
                  for t in ("entities", "relations", "aliases")}
    return counts


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
# The walk returns everything reachable in hop order, and on a real vault hop
# order is mostly plumbing. Measured before this existed, over 12 real questions:
# 77 facts recalled, 42 of them (54%) a hub or a grouping edge. `references`
# alone was 48% of 4,385 relations and nearly all of it an index or a log
# linking a note -- so a 3-hop walk from any seed reached the whole vault.
#
# So triples are scored rather than truncated in walk order. Nothing is dropped:
# a hub edge is the only fact some questions have, and demoting beats deleting
# when the alternative is an empty recall. It just loses to an edge that says
# something about the subject that was actually asked about.
PREDICATE_WEIGHT = {"related_to": 3.0, "cites": 2.0, "part_of": 1.5,
                    "references": 1.0, "in_domain": 0.4}
# Measured: median degree 6, p90 31, p95 46. Above 60 a node is on nearly every
# walk, which is exactly why it discriminates nothing.
HUB_DEGREE = 60
HUB_PENALTY = 0.25
# A wikilink in an append-only journal means "this was touched on this date",
# never "these two things are related" -- chronology, not topology. Degree alone
# does not catch it: rotating a log file drops the live log under HUB_DEGREE and
# its plumbing promptly comes back (measured 12% -> 18% of recalled facts).
JOURNAL_STEMS = ("log", "changelog", "journal", "daily")
JOURNAL_PENALTY = 0.25
# The question said this entity's name out loud. Has to outrank the whole
# predicate scale on its own, or topology beats relevance: at 2.5 a plain
# `references` edge naming the subject (1.0 x 2.5) still lost to an unrelated
# `related_to` (3.0), and a question about a slow model was answered with facts
# about an unrelated 3D globe.
NAMED_BONUS = 6.0
# Grouping edges say "these share a folder/domain", which is true of hundreds of
# notes. Two is enough context; the rest is filler.
GROUPING = ("in_domain",)
MAX_GROUPING = 2

STOP_WORDS = {"what", "which", "when", "where", "does", "do", "did", "is",
              "are", "was", "the", "a", "an", "and", "or", "for", "from",
              "with", "how", "why", "this", "that", "it", "to", "of", "in",
              "on", "my", "me", "i", "you", "we", "can", "should", "would"}


@dataclass
class Facts:
    triples: list  # (source, predicate, target, source_doc)
    notes: list    # (entity_name, description)
    ms: float

    def as_text(self) -> str:
        header = f"memory: {len(self.triples)} facts recalled in {self.ms:.0f} ms"
        if not self.triples:
            return header + "\n(no memory matches for this prompt)"
        width = max(len(f"{s} --[{p}]--> {t}") for s, p, t, _ in self.triples)
        lines = [f"{f'{s} --[{p}]--> {t}':<{width}}   ({doc})"
                 for s, p, t, doc in self.triples]
        text = header + "\n\n" + "\n".join(lines)
        if self.notes:
            # The conditions live on the entities, not the edges - carry them.
            text += "\n\nwhere:\n" + "\n".join(
                f"  {name}: {desc}" for name, desc in self.notes)
        return text


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= MIN_SEED_LEN and w not in STOP_WORDS}


def _is_journal(doc: str) -> bool:
    stem = doc.rsplit("/", 1)[-1].removesuffix(".md").lower()
    return stem in JOURNAL_STEMS or stem.startswith(
        tuple(f"{s}-" for s in JOURNAL_STEMS))


def _rank(triples: list, degrees: dict, nears: list, question: str) -> list:
    """Walk output as (score, triple), best first. Walk order breaks ties."""
    words = _words(question)
    scored = []
    for order, triple in enumerate(triples):
        src, pred, tgt, doc = triple
        score = PREDICATE_WEIGHT.get(pred, 1.0)
        if _is_journal(doc):
            score *= JOURNAL_PENALTY
        for name in (src, tgt):
            if degrees.get(name, 0) > HUB_DEGREE:
                score *= HUB_PENALTY
            if words & _words(name):
                score *= NAMED_BONUS
        # Relevance decays with distance from the seed: a fact two hops out is
        # about the subject's neighbours' neighbours, not the subject.
        scored.append((score / (1 + nears[order]), order, triple))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [(s, t) for s, _, t in scored]


def _seeds(db: sqlite3.Connection, question: str) -> list:
    """An entity is a seed when its name or an alias appears in the question.

    Upstream's rule, unchanged: exact, whole-word, and never wrong when it fires.
    """
    q = question.lower()
    found = {}
    rows = list(db.execute("SELECT id, name FROM entities"))
    rows += list(db.execute("SELECT entity_id, alias FROM aliases"))
    for eid, text in rows:
        if len(text) < MIN_SEED_LEN:
            continue
        if re.search(rf"\b{re.escape(text.lower())}\b", q):
            found[eid] = True
    return list(found)


# A whole-name match is exact and always right, and on a real question it is
# also usually empty: "why does knowledge compound" names no entity, so the walk
# has nowhere to start and recall returns nothing while the note sits in the
# graph. Word seeding closes that gap -- an entity whose name shares a
# significant word with the question is an entry point, and the ranking decides
# whether it deserved to be. Prefix matching, not stemming: "compound" reaches
# "Compounding Knowledge" without anyone shipping a stemmer.
MAX_WORD_SEEDS = 12


def _word_seeds(db: sqlite3.Connection, question: str, already: list) -> list:
    words = _words(question)
    if not words:
        return []
    rows = list(db.execute("SELECT id, name FROM entities"))
    rows += list(db.execute("SELECT entity_id, alias FROM aliases"))
    scored = []
    for eid, text in rows:
        if eid in already:
            continue
        name_words = _words(text)
        if not name_words:
            continue
        hits = sum(1 for w in name_words
                   if w in words or any(q.startswith(w) or w.startswith(q)
                                        for q in words if len(q) >= 4))
        if hits:
            # Prefer names the question matched more of: a two-word name fully
            # matched beats a five-word name brushed once.
            scored.append((hits / len(name_words), hits, eid))
    scored.sort(reverse=True)
    out, seen = [], set()
    for _ratio, _hits, eid in scored:
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
        if len(out) >= MAX_WORD_SEEDS:
            break
    return out


def recall(db_file: Path, question: str, hops: int = 3, top_k: int = 8) -> Facts:
    t0 = time.perf_counter()
    if not db_file.is_file():
        return Facts([], [], 0.0)
    # Pulled wider than top_k, then ranked and cut: the best fact is regularly
    # not in the first 8 rows of walk order.
    pull = max(top_k * 5, 40)
    with closing(sqlite3.connect(db_file)) as db:
        seeds = _seeds(db, question)
        seeds += _word_seeds(db, question, seeds)
        if not seeds:
            return Facts([], [], (time.perf_counter() - t0) * 1000)
        marks = ",".join("?" * len(seeds))
        rows = db.execute(WALK.format(seeds=marks), (*seeds, hops)).fetchall()
        triples = [(s, p, t, doc) for s, p, t, doc, _ in rows[:pull]]
        nears = [n or 0 for *_, n in rows[:pull]]
        degrees = {name: n for name, n in db.execute(DEGREE_SQL)}

        ranked, grouping = [], 0
        for _score, triple in _rank(triples, degrees, nears, question):
            if triple[1] in GROUPING:
                grouping += 1
                if grouping > MAX_GROUPING:
                    continue
            ranked.append(triple)
        ranked = ranked[:top_k]

        names = {n for s, _, t, _ in ranked for n in (s, t)}
        marks = ",".join("?" * len(names)) or "''"
        notes = db.execute(
            f"SELECT name, description FROM entities "  # noqa: S608
            f"WHERE name IN ({marks}) AND description != '' ORDER BY name",
            (*names,)).fetchall()
    return Facts(ranked, notes, (time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _vault(args) -> Path:
    raw = getattr(args, "vault", None) or os.environ.get("GRAPH_MEMORY_VAULT") or "."
    return Path(raw).expanduser().resolve()


def _db(vault: Path) -> Path:
    return vault / STATE_DIRNAME / DB_NAME


def cmd_build(args) -> int:
    vault = _vault(args)
    t0 = time.perf_counter()
    notes = read_notes(vault, args.scope)
    if not notes:
        print(f"no .md notes under {vault}"
              + (f" matching scope {args.scope!r}" if args.scope else ""))
        return 1
    nodes, edges, aliases = collect(notes)
    counts = write_db(_db(vault), nodes, edges, aliases)
    print(f"{len(notes)} notes -> {counts['entities']} entities, "
          f"{counts['relations']} relations, {counts['aliases']} aliases "
          f"in {(time.perf_counter() - t0):.1f}s")
    print(f"  {_db(vault)}")
    return 0


def cmd_recall(args) -> int:
    facts = recall(_db(_vault(args)), args.question, args.hops, args.top_k)
    print(facts.as_text())
    return 0


def cmd_hook(args) -> int:
    """Claude Code UserPromptSubmit hook: JSON on stdin, context on stdout.

    Never fails the turn. A hook that can crash the prompt is worse than a hook
    that occasionally recalls nothing, so everything here is wrapped.
    """
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        question = payload.get("prompt") or ""
        if not question.strip():
            return 0
        vault = Path(payload.get("cwd") or _vault(args))
        facts = recall(_db(vault), question, args.hops, args.top_k)
        if facts.triples:
            print("Knowledge-graph memory, walked from the notes' own "
                  "frontmatter and wikilinks (facts, not inference):\n")
            print(facts.as_text())
    except Exception:  # noqa: BLE001 - see docstring
        return 0
    return 0


def cmd_status(args) -> int:
    vault = _vault(args)
    db_file = _db(vault)
    if not db_file.is_file():
        print(f"no graph at {db_file}\n  build it: python graph_memory.py build")
        return 1
    with closing(sqlite3.connect(db_file)) as db:
        counts = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
                  for t in ("entities", "relations", "aliases")}
        top = db.execute(DEGREE_SQL + " ORDER BY 2 DESC LIMIT 5").fetchall()
    built = db_file.stat().st_mtime
    newest = max((p.stat().st_mtime for p in vault.rglob("*.md")
                  if not any(part in SKIP_DIRS for part in p.parts)), default=0)
    print(f"graph: {counts['entities']} entities, {counts['relations']} relations, "
          f"{counts['aliases']} aliases")
    print(f"built: {time.strftime('%Y-%m-%d %H:%M', time.localtime(built))}"
          f"  {'STALE - notes changed since' if newest > built else 'current'}")
    print("busiest nodes: " + ", ".join(f"{n} ({d})" for n, d in top))
    return 0


FIXTURE = {
    "concepts/Session Cache.md": (
        "---\ntype: concept\nrelated:\n  - \"[[LLM Wiki Pattern]]\"\n---\n"
        "A session cache is thrown away when the session ends.\n"
        "It also mentions [[Nonexistent Note]], which resolves to nothing.\n"),
    "concepts/LLM Wiki Pattern.md": (
        "---\ntype: concept\nsources:\n  - \"[[Karpathy Thread]]\"\n---\n"
        "The wiki is the durable half of the cache.\n"),
    "entities/Karpathy Thread.md": (
        "---\ntype: source\n---\nThe thread that named the pattern.\n"),
    "log.md": (
        "---\ntype: meta\n---\n"
        "2026-01-01 touched [[Session Cache]] and [[Karpathy Thread]].\n"),
}


def cmd_selftest(args) -> int:
    """Build a fixture, walk it, and assert the two things that actually break:
    a two-hop chain survives, and journal plumbing loses to a real edge."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp)
        for rel, text in FIXTURE.items():
            path = vault / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        notes = read_notes(vault, "")
        assert len(notes) == 4, f"read {len(notes)} notes, want 4"
        nodes, edges, aliases = collect(notes)
        counts = write_db(_db(vault), nodes, edges, aliases)
        assert counts["relations"] >= 4, counts

        # Two hops: Session Cache -> LLM Wiki Pattern -> Karpathy Thread. The
        # question never says "Karpathy", so only the walk can reach it.
        facts = recall(_db(vault), "what is the session cache", hops=3, top_k=8)
        rendered = facts.as_text()
        assert "Session Cache" in rendered, rendered
        assert "Karpathy Thread" in rendered, "two-hop chain lost:\n" + rendered

        # A dead link invents nothing.
        assert "Nonexistent" not in rendered

        # The journal's edges exist but must not lead.
        first = facts.triples[0]
        assert not _is_journal(first[3]), f"journal edge ranked first: {first}"

        # Word seeding: the question names no entity in full, and prefix
        # matching still has to find one. This is the case upstream returns
        # empty on, so it is the one most worth a test.
        partial = recall(_db(vault), "what gets cached in a session", top_k=5)
        assert partial.triples, "word seeding found nothing"
        assert "Session Cache" in partial.as_text(), partial.as_text()

        # ...without seeding on everything: an unrelated question stays empty.
        assert not recall(_db(vault), "what time is dinner").triples

        print(rendered)
        print("\nselftest: ok")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles are cp1252
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vault", help="vault root (default: $GRAPH_MEMORY_VAULT or .)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="notes -> .graph-memory/graph.db")
    b.add_argument("--scope", default="", help="only index paths under this prefix")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("recall", help="what the hook would inject")
    r.add_argument("question", nargs="+")
    r.add_argument("--hops", type=int, default=3)
    r.add_argument("--top-k", type=int, default=8)
    r.set_defaults(fn=cmd_recall)

    h = sub.add_parser("hook", help="UserPromptSubmit hook (JSON on stdin)")
    h.add_argument("--hops", type=int, default=3)
    h.add_argument("--top-k", type=int, default=8)
    h.set_defaults(fn=cmd_hook)

    sub.add_parser("status", help="counts and staleness").set_defaults(fn=cmd_status)
    sub.add_parser("selftest", help="build and walk a fixture").set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    if args.cmd == "recall":
        args.question = " ".join(args.question)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
