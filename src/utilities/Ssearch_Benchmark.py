"""Bounded, per-search plan selection. No persistent hardware decisions."""
from dataclasses import dataclass, field
import math
import time
import statistics
import threading


class SearchTiming:
    """Optional runner hook; start/stop bracket drained processing only."""
    def __init__(self):
        self.created = time.perf_counter()
        self.setup = self.processing = self.shutdown = 0.0

    def start(self):
        self.started = time.perf_counter()
        self.setup = self.started - self.created

    def stop(self):
        self.stopped = time.perf_counter()
        self.processing = self.stopped - self.started

    def finish(self):
        self.shutdown = time.perf_counter() - self.stopped

    def warm_executor(self, executor, count, callback=None):
        """Materialize every lazy executor thread before processing timing."""
        barrier = threading.Barrier(count)

        def warm():
            try:
                if callback is not None:
                    callback()
                barrier.wait(timeout=30)
            except BaseException:
                barrier.abort()
                raise

        futures = [executor.submit(warm) for _ in range(count)]
        for future in futures:
            future.result()


@dataclass
class SearchPlan:
    candidate: object
    variant: str
    precision: str
    lanes: int = 1
    observations: list = field(default_factory=list)
    results: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.candidate.spec, self.variant, self.precision, self.lanes)

    @property
    def execution(self):
        return (self.candidate, self.variant, self.precision, self.lanes)

    @property
    def label(self):
        return f"{self.candidate.display_name} / {self.variant} / {self.precision} / {self.lanes} lane(s)"

    def predicted(self, costs):
        remaining = sum(cost for index, cost in costs.items() if index not in self.results)
        if not remaining:
            return 0.0
        if not self.observations:
            return math.inf
        return statistics.median(
            timing.setup + timing.shutdown + rate * remaining
            for timing, rate in self.observations
        )


def stratified(tasks, lengths, count):
    ordered = sorted(tasks, key=lambda task: (lengths[int(task[0])], int(task[0])))
    count = min(len(ordered), max(0, int(count)))
    if count < 2:
        return ordered[:count]
    return [ordered[i * (len(ordered) - 1) // (count - 1)] for i in range(count)]


def rank_plans(plans, costs):
    remaining = [plan for plan in plans if plan.observations]
    ranked = []
    while remaining:
        best = min(plan.predicted(costs) for plan in remaining)
        eligible = [plan for plan in remaining if plan.predicted(costs) <= best * 1.05]
        # Stable ordering preserves discovery preference as the final tie breaker.
        winner = min(eligible, key=lambda plan: (
            plan.variant != "serial", plan.lanes, plan.variant == "tiled",
        ))
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


class SearchSelector:
    def __init__(self, tasks, lengths, query_length, execute, clock=None):
        self.tasks, self.lengths = tasks, lengths
        self.costs = {int(task[0]): max(1, query_length * lengths[int(task[0])]) for task in tasks}
        self.execute, self.clock = execute, clock or time.perf_counter
        self.started = self.clock()
        self.excluded = 0.0
        self.budget = 5.0
        self.plans = []
        self.messages = []
        self.sampled_ids = set()
        self.sample = stratified(tasks, lengths, 16)

    @property
    def elapsed(self):
        return max(0.0, self.clock() - self.started - self.excluded)

    def ranked(self):
        return rank_plans(self.plans, self.costs)

    def skip(self, plan, reason):
        message = f"{plan.label}: {reason}"
        self.messages.append(message)
        print(f"[Hardware] {message}")

    def allowed(self, plan):
        ranked = self.ranked()
        if self.elapsed >= self.budget:
            self.skip(plan, ("repeat skipped" if plan.observations else "unmeasured") + ": tuning budget exhausted")
            return False
        if not ranked:
            return True  # A usable forced plan must be measured even after slow initialization.
        observations = plan.observations or [
            obs for other in self.plans if other.candidate.spec == plan.candidate.spec
            for obs in other.observations
        ] or ranked[0].observations
        work = sum(self.costs[int(task[0])] for task in self.sample)
        estimate = statistics.median(t.setup + t.shutdown + rate * work for t, rate in observations)
        if plan.variant == "pool" and not plan.observations:
            # A serial runner's near-zero setup is not evidence for a fresh
            # spawned pool. Reserve a conservative first-use second; actual
            # process imports can still exceed this soft estimate.
            estimate = max(estimate, 1.0 + statistics.median(rate * work for _, rate in observations))
        if estimate > self.budget - self.elapsed or ranked[0].predicted(self.costs) <= 2 * estimate:
            self.skip(plan, ("repeat skipped" if plan.observations else "unmeasured") + ": estimated trial cost cannot repay tuning")
            return False
        return True

    def trial(self, plan, sample=None, required=False):
        if not required and not self.allowed(plan):
            return False
        sample = self.sample if sample is None else sample
        timing = SearchTiming()
        try:
            payload = self.execute(plan, sample, timing)
            expected = {int(task[0]) for task in sample}
            received = {int(row["index"]): row for row in payload}
            if len(payload) != len(received) or received.keys() != expected:
                raise ValueError("Search trial returned missing, duplicate, or unexpected targets")
            if not math.isfinite(timing.processing) or timing.processing < 0:
                raise ValueError("Invalid search timing")
            work = sum(self.costs[index] for index in expected)
            plan.observations.append((timing, max(timing.processing, 1e-9) / max(work, 1)))
            plan.results.update(received)
            self.sampled_ids.update(received)
            if plan not in self.plans:
                self.plans.append(plan)
            print(f"[Hardware] {plan.label}: setup={timing.setup:.3f}s; "
                  f"processing={timing.processing:.3f}s; shutdown={timing.shutdown:.3f}s; "
                  f"estimated remaining={plan.predicted(self.costs):.3f}s")
            return True
        except Exception as error:
            self.skip(plan, f"failed: {type(error).__name__}: {error}")
            return False

    def baseline(self, plan):
        if not self.trial(plan, stratified(self.tasks, self.lengths, 4), required=True):
            return False
        duration = plan.predicted(self.costs)
        timing, rate = plan.observations[0]
        total_duration = timing.setup + timing.shutdown + rate * sum(self.costs.values())
        self.budget = min(5.0, max(0.5, 0.05 * total_duration))
        mean_cost = sum(self.costs.values()) / len(self.costs)
        rate = plan.observations[0][1]
        count = min(64, max(16, math.ceil(0.1 / max(rate * mean_cost, 1e-9))))
        self.sample = stratified(self.tasks, self.lengths, count)
        return duration <= 1.0
