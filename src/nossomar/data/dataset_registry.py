"""Dataset registry skeleton by fidelity level for NOSSO-MAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    fidelity_level: str
    physics_class: str
    source_type: str
    source_tool_or_lab: str
    observable_keys: tuple[str, ...]
    notes: str = ''


class DatasetRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, DatasetRecord] = {}

    def register(self, record: DatasetRecord) -> None:
        if record.dataset_id in self._items:
            raise ValueError(f'Duplicate dataset_id: {record.dataset_id}')
        self._items[record.dataset_id] = record

    def get(self, dataset_id: str) -> DatasetRecord:
        return self._items[dataset_id]

    def list_by_fidelity(self, fidelity_level: str) -> tuple[DatasetRecord, ...]:
        return tuple(item for item in self._items.values() if item.fidelity_level == fidelity_level)


def build_default_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    registry.register(
        DatasetRecord(
            dataset_id='phase1_capytaine_dl2',
            fidelity_level='D-L2',
            physics_class='M1',
            source_type='numerical',
            source_tool_or_lab='Capytaine',
            observable_keys=('added_mass', 'radiation_damping', 'excitation_force', 'body_motion', 'absorbed_power'),
            notes='Current benchmark anchor for Phase 1 F1A.',
        )
    )
    return registry
