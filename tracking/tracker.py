from collections.abc import Callable, Sequence

from tracking.records import RunRecord


class ExperimentTracker:
    def __init__(self, maximise_metric: bool = True) -> None:
        self._records: list[RunRecord] = []
        self._maximise: bool = maximise_metric

    def log_run(self, record: RunRecord) -> None:
        self._records.append(record)

    def get_best_run(self) -> RunRecord | None:
        if not self._records:
            return None

        sorted_records: list[RunRecord] = sorted(
            self._records,
            key=lambda record: record.metrics.primary_score,
            reverse=self._maximise,
        )
        return sorted_records[0]

    def filter_runs(
        self, predicate: Callable[[RunRecord], bool]
    ) -> Sequence[RunRecord]:
        return [record for record in self._records if predicate(record)]

    def all_runs(self) -> Sequence[RunRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
