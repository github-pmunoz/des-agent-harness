from __future__ import annotations
import json
from dataclasses import dataclass, replace

from desh_chat.memory import Memory

# The model's knowledge graph, as a pluggable memory: subject-predicate-object triplets, one per
# (subject, predicate, object), in the order first written. What makes it a graph rather than a
# second scratchpad is that a subject and an object are entities — short names that recur across
# triplets — so the tools refuse prose in either place: given free text, a model files a paragraph
# under a filler subject and nothing connects.
MAX_ENTITY_CHARS = 80
MAX_PREDICATE_CHARS = 40

ONTOLOGY_PROMPT = (
    "The <ontology> is your knowledge graph: what you have established about the things the task "
    "is about, as subject-predicate-object triplets. A subject and an object are entities — a file, "
    "a function, a flag, a commit, a value — named the SAME way every time they appear, so that "
    "triplets about one entity connect; the predicate is the relation between them, a few words. "
    "Use ontology_write for each relation you establish and ontology_delete for one that turned out "
    "wrong. Never put a sentence in a triplet: an entity is a name, not a description."
)


def key_of(subject: str, predicate: str, object: str) -> str:
    """The dict key of a triplet: its three parts as a JSON list, which no part can forge."""
    return json.dumps([subject, predicate, object], ensure_ascii=False)


def too_long(subject: str, predicate: str, object: str) -> str | None:
    """The refusal for a part that is prose rather than a name, or None."""
    for part, text, limit in (("subject", subject, MAX_ENTITY_CHARS), ("predicate", predicate, MAX_PREDICATE_CHARS),
                              ("object", object, MAX_ENTITY_CHARS)):
        if len(text) > limit:
            return (f"not recorded: the {part} is {len(text)} characters, over {limit}. An entity is a short name and a "
                    f"predicate a few words; split the statement into triplets between named entities.")
    return None


def write(subject: str, predicate: str, object: str, ontology: dict[str, dict]) -> str:
    """Record a subject-predicate-object triplet in the ontology.

    Args:
        subject: The entity the triplet is about: a short name, the same every time.
        predicate: The relation between subject and object, in a few words.
        object: The entity or value the subject is related to: a short name or literal.
    """
    refusal = too_long(subject, predicate, object)
    if refusal is not None:
        return refusal
    key = key_of(subject, predicate, object)
    if key in ontology:
        return f"already recorded: {subject} -[{predicate}]-> {object}"
    ontology[key] = {"subject": subject, "predicate": predicate, "object": object}
    return f"recorded: {subject} -[{predicate}]-> {object}"


def delete(subject: str, predicate: str, object: str, ontology: dict[str, dict]) -> str:
    """Delete one triplet from the ontology.

    Args:
        subject: The subject of the triplet to delete.
        predicate: Its predicate.
        object: Its object.
    """
    key = key_of(subject, predicate, object)
    if key in ontology:
        del ontology[key]
        return f"deleted: {subject} -[{predicate}]-> {object}"
    return f"not found: {subject} -[{predicate}]-> {object}"


def clear(ontology: dict[str, dict]) -> str:
    """Clear the ontology: remove every recorded triplet."""
    n = len(ontology)
    ontology.clear()
    return f"cleared {n} triplet{'s' if n != 1 else ''}" if n else "already empty"


@dataclass(frozen=True)
class Triplet:
    """One ontology entry: the subject, the predicate (relation) and the object."""
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class Ontology:
    """The knowledge graph, as a value: ordered triplets, one per (subject, predicate, object)."""
    triplets: tuple[Triplet, ...] = ()

    def with_triplet(self, subject: str, predicate: str, object: str) -> Ontology:
        """Record a triplet, keeping the first written order: an existing triplet is not moved."""
        t = Triplet(subject, predicate, object)
        return self if t in self.triplets else replace(self, triplets=self.triplets + (t,))

    @classmethod
    def from_dict(cls, d: dict[str, dict]) -> Ontology:
        return cls(tuple(Triplet(t["subject"], t["predicate"], t["object"]) for t in d.values()))

    def to_dict(self) -> dict[str, dict]:
        """The dict the tools work on and the session file stores: keyed by the triplet, in order."""
        return {key_of(t.subject, t.predicate, t.object): {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in self.triplets}

    def render(self) -> str:
        """One `subject -[predicate]-> object` line per triplet, in order (MemoryValue.render)."""
        return "\n".join(f"{t.subject} -[{t.predicate}]-> {t.object}" for t in self.triplets)


# A subagent gets none: its graph dies with it, and nothing hands it back to the parent.
ONTOLOGY = Memory(
    name="ontology",
    empty=Ontology,
    from_dict=Ontology.from_dict,
    tools=((write, {"name": "ontology_write", "target": "subject"}),
           (delete, {"name": "ontology_delete", "target": "subject"}),
           (clear, {"name": "ontology_clear"})),
    prompt=ONTOLOGY_PROMPT,
    subagent="off",
)
