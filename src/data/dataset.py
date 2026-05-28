from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

TripleText = Tuple[str, str, str]
TripleIds = Tuple[int, int, int]


@dataclass
class KGDataset:
    dataset_path: Path
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)


def _resolve_split_path(dataset_path: Path, split_name: str) -> Path:
    direct = dataset_path / f"{split_name}.txt"
    nested = dataset_path / "data" / f"{split_name}.txt"

    if direct.exists():
        return direct
    if nested.exists():
        return nested

    raise FileNotFoundError(
        f"Could not find {split_name}.txt in either {direct} or {nested}."
    )


def _read_triples(path: Path) -> List[TripleText]:
    triples: List[TripleText] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def _build_vocab(
    train: Sequence[TripleText],
    valid: Sequence[TripleText],
    test: Sequence[TripleText],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    entities = set()
    relations = set()

    for h, r, t in [*train, *valid, *test]:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity_to_id = {entity: i for i, entity in enumerate(sorted(entities))}

    base_relations = sorted(relations)
    relation_to_id = {relation: i for i, relation in enumerate(base_relations)}

    num_orig_relations = len(base_relations)
    for i, relation in enumerate(base_relations):
        # Generate inverse relation tokens and assign offset IDs
        relation_to_id[f"{relation}_inverse"] = i + num_orig_relations

    return entity_to_id, relation_to_id


def _to_ids(
    triples: Sequence[TripleText],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    add_inverses: bool = False,
) -> torch.Tensor:
    encoded: List[TripleIds] = []

    num_orig_relations = len(relation_to_id) // 2

    for h, r, t in triples:
        h_id, r_id, t_id = entity_to_id[h], relation_to_id[r], entity_to_id[t]
        encoded.append((h_id, r_id, t_id))
        if add_inverses:
            encoded.append((t_id, r_id + num_orig_relations, h_id))

    return torch.tensor(encoded, dtype=torch.long)


def load_dataset(dataset_path: str) -> KGDataset:
    """Load a knowledge graph dataset from a given path.

    Args:
        dataset_path (str): The file path to the dataset directory containing 'train.txt', 'valid.txt', and 'test.txt'.

    Returns:
        KGDataset: A dataclass instance containing the loaded tensor splits and vocabularies.
    """
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    train_path = _resolve_split_path(root, "train")
    valid_path = _resolve_split_path(root, "valid")
    test_path = _resolve_split_path(root, "test")

    train_text = _read_triples(train_path)
    valid_text = _read_triples(valid_path)
    test_text = _read_triples(test_path)

    entity_to_id, relation_to_id = _build_vocab(train_text, valid_text, test_text)

    train = _to_ids(train_text, entity_to_id, relation_to_id, add_inverses=False)
    valid = _to_ids(valid_text, entity_to_id, relation_to_id, add_inverses=False)
    test = _to_ids(test_text, entity_to_id, relation_to_id, add_inverses=False)

    return KGDataset(
        dataset_path=root,
        train=train,
        valid=valid,
        test=test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
