"""Grounding by ablation.

The claim glasshouse makes is causal, not correlational. Everyone else asks a
model whether an answer is supported, or checks whether a sentence looks like
a retrieved chunk. Both measure resemblance. Neither can distinguish a sentence
the model read from the evidence from a sentence it already knew and would have
produced from an empty context.

Ablation asks the counterfactual directly: **take the evidence away and see
what changes.** If withholding chunk 3 makes a sentence disappear or change,
chunk 3 was load-bearing for that sentence. If withholding every chunk changes
nothing, the sentence never came from the corpus at all.

Three things make that harder than it sounds, and each has a mechanism here:

**The answer moves for reasons unrelated to evidence.** Generation at
temperature 0 is not deterministic across different prompts, and removing a
chunk changes the prompt. So a *control* run sees exactly the same evidence in
a different order, and the movement it produces becomes the noise floor every
effect has to clear.

**Redundant evidence hides itself from leave-one-out.** Two chunks saying the
same thing means removing either alone changes nothing, and a well-grounded
claim looks unsupported. So claims that survive every single removal go to a
*coalition* search that removes likely groups together. Retrieval already
fights this with MMR (see :mod:`glasshouse.index`); this catches the rest.

**Cost is quadratic if you are careless.** Every extra chunk is another
generation. Runs are budgeted, deduplicated through the cassette, and fanned
out concurrently, and what got skipped is reported rather than hidden.

What this cannot do is in the README, under "What this does not prove". The
short version: a grounded verdict says the model *used* the chunk, not that the
chunk is *true*, and not that the sentence is a correct reading of it.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .events import Emitter, emit
from .generate import build_prompt, system_for
from .llm import DEFAULT_MODEL, LLM, Request
from .models import (
    Chunk,
    ClaimVerdict,
    Report,
    Retrieved,
    Run,
    RunKind,
    Support,
    Usage,
    Verdict,
    ZERO_USAGE,
)
from .similarity import Matcher
from .text import carries_a_claim, sentences


@dataclass(frozen=True)
class AblationPolicy:
    """Thresholds and budgets.

    Candidate support thresholds are evaluated against the committed independent
    sentence-label fixture, which exposes precision, recall, false-positive
    rate, and false-negative rate across a grid.
    """

    #: How much similarity a sentence must lose, above the noise floor, before
    #: a chunk is credited with supporting it.
    support_threshold: float = 0.12
    #: How closely a sentence must match the no-evidence answer before it is
    #: called model memory.
    memory_threshold: float = 0.78
    #: A second chunk is co-credited if its effect is at least this fraction
    #: of the strongest one's.
    share: float = 0.6
    #: Hard ceiling on generations for one question, control and closed-book
    #: included. Reached means some verdicts come back UNDETERMINED, which is
    #: reported rather than quietly rounded to "unsupported".
    max_runs: int = 28
    #: Runs that see identical evidence in a different order.
    control_runs: int = 1
    #: Largest group of chunks removed together in the coalition search.
    coalition_size: int = 3
    #: How many chunks per unresolved claim are considered for coalitions.
    coalition_candidates: int = 3
    concurrency: int = 6
    model: str = DEFAULT_MODEL
    max_tokens: int = 600

    def __post_init__(self) -> None:
        if not 0.0 < self.support_threshold < 1.0:
            raise ValueError("support_threshold must be between 0 and 1")
        if not 0.0 < self.memory_threshold <= 1.0:
            raise ValueError("memory_threshold must be between 0 and 1")
        if self.max_runs < 2:
            raise ValueError("max_runs must leave room for at least a control")
        if self.coalition_size < 2:
            raise ValueError("a coalition of one is a leave-one-out run")


class Budget:
    """A countdown on generations."""

    def __init__(self, limit: int):
        self.limit = limit
        self.spent = 0

    def take(self, count: int = 1) -> bool:
        if self.spent + count > self.limit:
            return False
        self.spent += count
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)


class Ablator:
    """Runs the ablation sweep for one question and scores the result."""

    def __init__(
        self,
        llm: LLM,
        matcher: Matcher,
        policy: AblationPolicy | None = None,
        emitter: Emitter | None = None,
    ):
        self.llm = llm
        self.matcher = matcher
        self.policy = policy or AblationPolicy()
        self.emitter = emitter
        self._gate = asyncio.Semaphore(self.policy.concurrency)

    # -- entry point --------------------------------------------------------

    async def analyse(self, question: str, retrieved: Sequence[Retrieved]) -> Report:
        started = time.perf_counter()
        chunks = [r.chunk for r in retrieved]
        budget = Budget(self.policy.max_runs)
        runs: list[Run] = []

        budget.take()
        full = await self._generate("full", RunKind.FULL, question, chunks, ())
        runs.append(full)
        claims = list(full.sentences)

        await emit(
            self.emitter,
            "answer",
            answer=full.answer,
            claims=claims,
            checkable=[carries_a_claim(c) for c in claims],
        )

        if not claims:
            return self._report(question, full, retrieved, (), runs, started, False)

        planned = self._plan(chunks)
        allowed = planned[: budget.remaining]
        budget.take(len(allowed))
        truncated = len(allowed) < len(planned)

        results = await self._run_all(question, chunks, allowed)
        runs.extend(results)

        survival = await asyncio.to_thread(self._score_survival, claims, results)
        noise = self._noise_floor(claims, results, survival)
        memory = self._memory(claims, results, survival)

        verdicts = self._first_pass(claims, chunks, results, survival, noise, memory)

        unresolved = [
            v.index for v in verdicts if v.verdict is Verdict.UNDETERMINED
        ]
        if unresolved:
            coalitions = await asyncio.to_thread(
                self._plan_coalitions, unresolved, verdicts, chunks
            )
            affordable = coalitions[: budget.remaining]
            budget.take(len(affordable))
            truncated = truncated or len(affordable) < len(coalitions)

            if affordable:
                extra = await self._run_all(question, chunks, affordable)
                runs.extend(extra)
                survival.update(
                    await asyncio.to_thread(self._score_survival, claims, extra)
                )
                verdicts = self._coalition_pass(
                    verdicts, unresolved, extra, survival, noise, memory
                )

        verdicts = tuple(self._settle(v, truncated) for v in verdicts)
        await emit(
            self.emitter,
            "verdicts",
            claims=[_verdict_payload(v) for v in verdicts],
        )
        return self._report(
            question, full, retrieved, verdicts, runs, started, truncated
        )

    # -- planning -----------------------------------------------------------

    def _plan(self, chunks: Sequence[Chunk]) -> list[tuple[str, RunKind, tuple[str, ...]]]:
        """The runs to fire after the full answer, in priority order.

        Ordering matters when the budget bites: the closed-book run is the one
        that distinguishes "grounded" from "the model already knew this", so it
        goes first and is never the run that gets dropped.
        """
        plan: list[tuple[str, RunKind, tuple[str, ...]]] = [
            ("closed", RunKind.CLOSED_BOOK, tuple(c.chunk_id for c in chunks))
        ]
        if len(chunks) > 1:
            for i in range(self.policy.control_runs):
                plan.append(("control-%d" % i, RunKind.CONTROL, ()))
        for chunk in chunks:
            plan.append(("loo-%s" % chunk.chunk_id, RunKind.LEAVE_ONE_OUT, (chunk.chunk_id,)))
        return plan

    def _plan_coalitions(
        self,
        unresolved: Sequence[int],
        verdicts: Sequence[ClaimVerdict],
        chunks: Sequence[Chunk],
    ) -> list[tuple[str, RunKind, tuple[str, ...]]]:
        """Groups worth removing together.

        Exhaustive subsets are ``2^k``. Instead, for each unresolved claim take
        the handful of chunks that most resemble it -- redundant support has to
        come from chunks that actually say the thing -- and remove their pairs
        and triples. Groups are deduplicated across claims because one run
        answers the question for every claim at once.
        """
        texts = [c.text for c in chunks]
        seen: set[tuple[str, ...]] = set()
        groups: list[tuple[str, ...]] = []

        for index in unresolved:
            claim = verdicts[index]
            ranked = self.matcher.rank_by_similarity(claim.text, texts)
            candidates = [
                chunks[i].chunk_id
                for i, _ in ranked[: self.policy.coalition_candidates]
            ]
            for size in range(2, min(self.policy.coalition_size, len(candidates)) + 1):
                for group in itertools.combinations(candidates, size):
                    key = tuple(sorted(group))
                    if key not in seen:
                        seen.add(key)
                        groups.append(key)

        # Smaller groups first: they attribute more precisely, and if the
        # budget runs out the coarse ones are the right ones to lose.
        groups.sort(key=len)
        return [
            ("coalition-%d" % n, RunKind.COALITION, group)
            for n, group in enumerate(groups)
        ]

    # -- execution ----------------------------------------------------------

    async def _run_all(
        self,
        question: str,
        chunks: Sequence[Chunk],
        plan: Sequence[tuple[str, RunKind, tuple[str, ...]]],
    ) -> list[Run]:
        tasks = [
            self._guarded(run_id, kind, question, chunks, removed)
            for run_id, kind, removed in plan
        ]
        return list(await asyncio.gather(*tasks))

    async def _guarded(self, run_id, kind, question, chunks, removed) -> Run:
        async with self._gate:
            run = await self._generate(run_id, kind, question, chunks, removed)
        await emit(
            self.emitter,
            "run",
            run_id=run.run_id,
            kind=run.kind.value,
            removed=list(run.removed),
            answer=run.answer,
            cost_usd=run.usage.cost_usd,
            input_tokens=run.usage.input_tokens,
            output_tokens=run.usage.output_tokens,
        )
        return run

    async def _generate(
        self,
        run_id: str,
        kind: RunKind,
        question: str,
        chunks: Sequence[Chunk],
        removed: tuple[str, ...],
    ) -> Run:
        withheld = set(removed)
        visible = [c for c in chunks if c.chunk_id not in withheld]
        if kind is RunKind.CONTROL:
            visible = _rotate(list(chunks))

        request = Request(
            system=system_for(visible),
            prompt=build_prompt(question, visible),
            model=self.policy.model,
            max_tokens=self.policy.max_tokens,
        )
        completion = await self.llm.complete(request)
        return Run(
            run_id=run_id,
            kind=kind,
            removed=removed,
            answer=completion.text,
            sentences=tuple(sentences(completion.text)),
            usage=completion.usage,
        )

    # -- scoring ------------------------------------------------------------

    def _survival(self, claims: Sequence[str], run: Run) -> np.ndarray:
        return self.matcher.survival(claims, run.sentences)

    def _score_survival(
        self, claims: Sequence[str], runs: Sequence[Run]
    ) -> dict[str, np.ndarray]:
        return {run.run_id: self._survival(claims, run) for run in runs}

    def _noise_floor(
        self,
        claims: Sequence[str],
        runs: Sequence[Run],
        survival: dict[str, np.ndarray],
    ) -> np.ndarray:
        """How much each sentence moves when nothing was taken away.

        With no control run available -- a single-chunk corpus -- the floor is
        zero, which makes every effect look real. That is the honest default:
        with one chunk there is nothing to be confused about.
        """
        controls = [r for r in runs if r.kind is RunKind.CONTROL]
        if not controls:
            return np.zeros(len(claims), dtype=np.float32)
        drops = np.stack([1.0 - survival[r.run_id] for r in controls])
        # Median rather than mean: one control that wandered off should not
        # raise the bar for every sentence in the answer.
        return np.clip(np.median(drops, axis=0), 0.0, 0.9).astype(np.float32)

    def _memory(
        self,
        claims: Sequence[str],
        runs: Sequence[Run],
        survival: dict[str, np.ndarray],
    ) -> np.ndarray:
        for run in runs:
            if run.kind is RunKind.CLOSED_BOOK:
                return survival[run.run_id]
        return np.zeros(len(claims), dtype=np.float32)

    def _first_pass(
        self,
        claims: Sequence[str],
        chunks: Sequence[Chunk],
        runs: Sequence[Run],
        survival: dict[str, np.ndarray],
        noise: np.ndarray,
        memory: np.ndarray,
    ) -> list[ClaimVerdict]:
        loo = {
            run.removed[0]: survival[run.run_id]
            for run in runs
            if run.kind is RunKind.LEAVE_ONE_OUT and run.removed
        }

        verdicts: list[ClaimVerdict] = []
        for index, text in enumerate(claims):
            if not carries_a_claim(text):
                verdicts.append(
                    ClaimVerdict(
                        index=index,
                        text=text,
                        verdict=Verdict.NO_CLAIM,
                        noise_floor=float(noise[index]),
                        memory=float(memory[index]),
                        note="no checkable claim",
                    )
                )
                continue

            supports = []
            for chunk in chunks:
                if chunk.chunk_id not in loo:
                    continue
                raw = float(1.0 - loo[chunk.chunk_id][index])
                supports.append(
                    Support(
                        chunk_id=chunk.chunk_id,
                        effect=max(0.0, raw - float(noise[index])),
                        raw_drop=raw,
                    )
                )
            supports.sort(key=lambda s: -s.effect)

            verdicts.append(
                self._decide(index, text, supports, float(noise[index]), float(memory[index]))
            )
        return verdicts

    def _decide(
        self,
        index: int,
        text: str,
        supports: list[Support],
        noise: float,
        memory: float,
    ) -> ClaimVerdict:
        best = supports[0].effect if supports else 0.0

        if best >= self.policy.support_threshold:
            floor = max(self.policy.support_threshold, self.policy.share * best)
            credited = tuple(
                replace(s, credited=s.effect >= floor) for s in supports
            )
            names = [s.chunk_id for s in credited if s.credited]
            return ClaimVerdict(
                index=index,
                text=text,
                verdict=Verdict.GROUNDED,
                support=credited,
                noise_floor=noise,
                memory=memory,
                note="withholding %s changed this sentence (%.2f over a %.2f noise floor)"
                % (_join(names), best, noise),
            )

        if memory >= self.policy.memory_threshold:
            return ClaimVerdict(
                index=index,
                text=text,
                verdict=Verdict.MODEL_MEMORY,
                support=tuple(supports),
                noise_floor=noise,
                memory=memory,
                note="the model says this with no documents at all (%.2f match), "
                "so it did not come from your corpus" % memory,
            )

        # Neither the evidence nor the model's memory explains it yet. The
        # coalition search gets a turn before this becomes a finding.
        return ClaimVerdict(
            index=index,
            text=text,
            verdict=Verdict.UNDETERMINED,
            support=tuple(supports),
            noise_floor=noise,
            memory=memory,
            note="no single chunk mattered",
        )

    def _coalition_pass(
        self,
        verdicts: list[ClaimVerdict],
        unresolved: Sequence[int],
        runs: Sequence[Run],
        survival: dict[str, np.ndarray],
        noise: np.ndarray,
        memory: np.ndarray,
    ) -> list[ClaimVerdict]:
        out = list(verdicts)
        for index in unresolved:
            claim = out[index]
            best_effect = 0.0
            best_group: tuple[str, ...] = ()
            for run in runs:
                raw = float(1.0 - survival[run.run_id][index])
                effect = max(0.0, raw - float(noise[index]))
                if effect > best_effect:
                    best_effect, best_group = effect, run.removed

            if best_effect < self.policy.support_threshold:
                continue

            members = set(best_group)
            support = tuple(
                replace(s, credited=s.chunk_id in members, joint=s.chunk_id in members)
                for s in claim.support
            )
            out[index] = replace(
                claim,
                verdict=Verdict.GROUNDED,
                support=support,
                note="no chunk mattered alone, but removing %s together changed "
                "this sentence (%.2f) -- they say the same thing"
                % (_join(sorted(members)), best_effect),
            )
        return out

    def _settle(self, verdict: ClaimVerdict, truncated: bool) -> ClaimVerdict:
        """Turn anything still undetermined into a final answer."""
        if verdict.verdict is not Verdict.UNDETERMINED:
            return verdict
        if truncated:
            return replace(
                verdict,
                note="ran out of the run budget before this could be resolved",
            )
        return replace(
            verdict,
            verdict=Verdict.UNSUPPORTED,
            note="no evidence mattered to this sentence and the model does not "
            "assert it from memory -- most often connective wording, "
            "occasionally a claim we could not trace",
        )

    def _report(
        self,
        question: str,
        full: Run,
        retrieved: Sequence[Retrieved],
        claims: Sequence[ClaimVerdict],
        runs: Sequence[Run],
        started: float,
        truncated: bool,
    ) -> Report:
        total = ZERO_USAGE
        for run in runs:
            total = total + run.usage
        return Report(
            question=question,
            answer=full.answer,
            retrieved=tuple(retrieved),
            claims=tuple(claims),
            runs=tuple(runs),
            usage=Usage(
                input_tokens=total.input_tokens,
                output_tokens=total.output_tokens,
                cost_usd=total.cost_usd,
                cached=all(r.usage.cached for r in runs) if runs else False,
            ),
            elapsed_s=time.perf_counter() - started,
            truncated=truncated,
        )


def _rotate(chunks: list[Chunk]) -> list[Chunk]:
    """Reorder evidence without changing it.

    A rotation by half the list moves every chunk while keeping local
    adjacency, which is a milder perturbation than a reversal and therefore a
    more representative estimate of ordinary order sensitivity.
    """
    if len(chunks) < 2:
        return chunks
    pivot = len(chunks) // 2
    return chunks[pivot:] + chunks[:pivot]


def _join(names: Sequence[str]) -> str:
    names = list(names)
    if not names:
        return "nothing"
    if len(names) == 1:
        return names[0]
    return "%s and %s" % (", ".join(names[:-1]), names[-1])


def _verdict_payload(verdict: ClaimVerdict) -> dict:
    return {
        "index": verdict.index,
        "text": verdict.text,
        "verdict": verdict.verdict.value,
        "note": verdict.note,
        "noise_floor": round(verdict.noise_floor, 4),
        "memory": round(verdict.memory, 4),
        "support": [
            {
                "chunk_id": s.chunk_id,
                "effect": round(s.effect, 4),
                "raw_drop": round(s.raw_drop, 4),
                "credited": s.credited,
                "joint": s.joint,
            }
            for s in verdict.support
        ],
    }
