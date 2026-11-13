"""Progress reports and the cancel request inside the steps of a solve.

A one-argument callback receives a Progress for the stages (the statics first, then the
assembly, the LU and the outputs of each frequency), at most every 0.1 s, and for every finished
frequency; a two-argument callback receives (done, total) after each frequency. Returning False
in the LU of the first frequency cancels the solve before it ends; returned from the two-argument
form it cancels between frequencies, and an exception raised in a callback ends the solve with it.
A callback does not change the results; the model refuses its changes and queries while it
calculates. Every frequency runs on a frequency worker of a fixed number (0 to f - 1); the phases
come in their order; the second-order steps (the quadratic QTF's rows, the sub-problems, the outer
integral and the pairs of the potential QTF) report their items as the frequencies, and a cancel
inside them stops the solve within a second. Meshes are procedural (tests/python/meshes.py)."""

import threading
import time

import pytest

import hydrobem as hb

from bodies import coarse_cylinder, slow_cylinder
from problems import flat_surface, model
from regression_problems import outputs


def _model(f=1, i=2):
    """2048 panels without symmetry: a solve of seconds, so that the stages of a frequency report
    (they report at most every 0.1 s) and a poll from another thread sees the solve running. The
    thread counts are fixed only where a test is about them: they set the order of the reports and
    how many frequencies run at once."""
    threads = {"frequency_threads": f, "inner_threads": i}
    return model({"mesh": slow_cylinder()}, [0.0, 1.5, 2.0], [0.0], options=threads)


def _small(omegas=(0.0, 1.5, 2.0), **kw):
    """32 panels of a quarter cylinder: for the tests that count frequencies, not stages."""
    return model({"mesh": coarse_cylinder()}, list(omegas), [0.0], **kw)


def test_the_stages_are_reported():
    reports = []
    r = _model().solve(reports.append)
    assert r.added_mass is not None
    stages = [p.stage for p in reports]
    assert set(stages) <= {"statics", "assembly", "lu", "outputs", "frequency_done", "phase", "worker_idle"}
    assert stages[0] == "statics" and reports[0].frequency is None
    done = [p for p in reports if p.stage == "frequency_done"]
    assert [p.stage_done for p in done] == [1, 2, 3] and all(p.stage_total == 3 for p in done)
    assert all(p.frequencies_total == 3 for p in reports)
    assert {p.frequency for p in done} == {-1, 0, 1}  # the limit omega = 0, then the two frequencies
    # the stage reports come at most every 0.1 s (the end of a frequency and the phases always)
    times = [p.seconds for p in reports if p.stage not in ("frequency_done", "phase")]
    assert all(b - a >= 0.099 for a, b in zip(times, times[1:], strict=False))
    assert all(0 <= p.stage_done <= p.stage_total for p in reports)


def _second_order(panel_size=0.5, f=2, i=1):
    """The fixed coarse cylinder with the quadratic QTF and both methods of the potential QTF: the
    second-order steps after its two frequencies."""
    m = model({"mesh": coarse_cylinder(), "modes": []}, [1.2, 1.6], [0.0], environment={"depth": 3.0},
              outputs={"qtf_potential_direct": True, "qtf_potential_indirect": True,
                       "qtf_quadratic_pressure_integration": True},
              options={"frequency_threads": f, "inner_threads": i})  # fmt: skip
    m.second_order.update(**flat_surface("free_surface", {"inner_radius": 2.5, "panel_size": panel_size}))
    return m


def _polled(m):
    """The solve of m on another thread, polled every 10 ms: the polls and the end."""
    t = threading.Thread(target=m.solve)
    t.start()
    polls = []
    while t.is_alive():
        polls.append(m.progress())
        time.sleep(0.01)
    t.join()
    return polls, m.progress()


STEPS = ["quadratic_pressure_integration", "potential_first_order", "potential_radiation", "potential_fits",
         "potential_outer", "potential_pairs", "potential_direct"]  # fmt: skip
PHASES = ["preparation", "limits", "first_order", "second_order", "finishing", "done"]


def test_the_second_order_reports_its_steps():
    # the model's frequencies count as before (each done once, the totals unchanged); the steps of
    # the second order follow them, their items with the stages of a frequency (the sub-problems)
    # or "second_order", every item on a worker of the model's two (or on all: worker None)
    m = _second_order()
    reports = []
    r = m.solve(reports.append)
    assert r.qtf.potential.direct.sum is not None and r.qtf.potential.indirect.sum is not None
    assert all(p.frequencies_total == 2 and p.frequencies_done <= 2 for p in reports)
    assert [p.stage_done for p in reports if p.stage == "frequency_done"] == [1, 2]
    assert all(p.worker in (None, 0, 1) for p in reports)
    # the second order only after both frequencies
    phases = [p.phase for p in reports if p.stage == "phase"]
    assert phases == ["first_order", "second_order", "finishing", "done"]
    first = next(k for k, p in enumerate(reports) if p.step is not None)
    assert sum(p.stage == "frequency_done" for p in reports[:first]) == 2
    assert {p.step for p in reports if p.step is not None} <= set(STEPS)
    end = m.progress()
    assert not end.running and end.done == end.total == 2 and end.phase == "done"
    assert all(f.state == "done" and f.worker in (0, 1) for f in end.frequencies)
    assert [s.step for s in end.second_order] == STEPS
    for s in end.second_order:
        assert s.state == "done" and s.fraction == 1.0 and s.items, s
        assert all(i.state == "done" and i.fraction == 1.0 and i.worker in (None, 0, 1) for i in s.items), s
    # the rows of the quadratic QTF, the sub-problems' frequencies and the pair frequencies run on
    # the workers; the fits and the batches of the outer integral on all of them
    by = {s.step: s for s in end.second_order}
    assert len(by["quadratic_pressure_integration"].items) == 2
    assert len(by["potential_first_order"].items) == 2
    for k in ("quadratic_pressure_integration", "potential_first_order", "potential_radiation", "potential_pairs",
              "potential_direct"):  # fmt: skip
        assert all(i.worker in (0, 1) for i in by[k].items), k
    assert all(i.worker is None for k in ("potential_fits", "potential_outer") for i in by[k].items)
    assert [w.state for w in end.workers] == ["done", "done"]


def test_the_second_order_progresses_monotonically():
    # polled while it runs: the steps begin in their order, their fractions and those of their
    # items never decrease, a worker is on one item at a time
    m = _second_order(panel_size=0.3)
    polls, end = _polled(m)
    assert end.phase == "done" and all(s.state == "done" for s in end.second_order)
    seen = [p for p in polls if p.second_order]
    assert seen, "the second order was not seen running"
    for a, b in zip(polls, polls[1:], strict=False):
        assert PHASES.index(b.phase) >= PHASES.index(a.phase)
        assert [s.step for s in b.second_order[: len(a.second_order)]] == [s.step for s in a.second_order]
        for sa, sb in zip(a.second_order, b.second_order, strict=False):
            assert sb.fraction >= sa.fraction
            for ia, ib in zip(sa.items, sb.items, strict=True):
                assert ib.fraction >= ia.fraction
    for p in seen:
        assert p.phase in ("second_order", "finishing", "done")
        for s in p.second_order:
            assert all(i.worker in (None, 0, 1) for i in s.items)
        running = [w for w in p.workers if w.state == "running" and w.index is not None]
        assert all(w.task in STEPS for w in running if p.phase == "second_order")


def test_workers_phases_and_idle_workers():
    # two frequency workers for the limit and three frequencies of seconds each: every frequency
    # on worker 0 or 1, two running at once on different workers, a worker's frequencies one
    # after the other; at the end one worker idle while the other runs the last frequency; the
    # phases in their order, every worker done after the solve
    m = model({"mesh": slow_cylinder()}, [0.0, 1.0, 1.5, 2.0], [0.0],
              options={"frequency_threads": 2, "inner_threads": 1})  # fmt: skip
    reports = []
    t = threading.Thread(target=m.solve, args=(reports.append,))
    t.start()
    polls = []
    while t.is_alive():
        polls.append(m.progress())
        time.sleep(0.01)
    t.join()
    end = m.progress()
    assert end.phase == "done" and [w.state for w in end.workers] == ["done", "done"]
    assert all(w.task is None for w in end.workers)
    assert all(f.state == "done" and f.worker in (0, 1) for f in end.frequencies)
    assert end.limits[0].progress.worker == 0  # the limits on the first worker
    # the phases: reported (always) and polled in their order
    phases = [p.phase for p in reports if p.stage == "phase"]
    assert phases == ["limits", "first_order", "finishing", "done"]
    polled = [p.phase for p in polls if p.running]
    assert all(PHASES.index(b) >= PHASES.index(a) for a, b in zip(polled, polled[1:], strict=False))
    assert {"limits", "first_order"} <= set(polled)
    # the reports name their worker
    assert all(p.worker in (0, 1) for p in reports if p.stage in ("assembly", "lu", "outputs", "frequency_done"))
    done = [p for p in reports if p.stage == "frequency_done" and p.frequency >= 0]
    assert all(p.worker == end.frequencies[p.frequency].worker for p in done)
    # in every poll: a running frequency has a worker, two at once have different ones
    for p in polls:
        run = [(k, f.worker) for k, f in enumerate(p.frequencies) if f.state == "running"]
        assert all(w in (0, 1) for _, w in run)
        assert len({w for _, w in run}) == len(run)
    assert any(len([f for f in p.frequencies if f.state == "running"]) == 2 for p in polls)
    # a worker's frequencies do not overlap: each began after the one before it on that worker ended
    for w in (0, 1):
        mine = [k for k, f in enumerate(end.frequencies) if f.worker == w]
        for p in polls:
            assert sum(p.frequencies[k].state == "running" for k in mine) <= 1
    # three frequencies on two workers: one is idle at the end of the first order
    assert any(p.phase == "first_order" and sorted(w.state for w in p.workers) == ["idle", "running"]
               for p in polls)  # fmt: skip


def test_the_two_argument_form_counts_cancels_and_raises():
    # (done, total) after each frequency in the order they end, also with two frequency threads
    seen = []
    _small(options={"frequency_threads": 2}).solve(lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
    # False stops the solve between frequencies
    seen = []
    p = _small()
    with pytest.raises(hb.SolveCancelledError, match="cancel"):
        p.solve(lambda done, total: seen.append(done) or False)
    assert seen and len(seen) < 3

    def fails(done, total):
        raise KeyError("from the callback")

    with pytest.raises(KeyError, match="from the callback"):
        p.solve(fails)
    assert p.solve(lambda done, total: None).added_mass is not None


def test_a_cancel_inside_a_frequency():
    # the first report inside a frequency (the assembly reports on every machine) stops the solve
    # there: no frequency is done after it
    seen = []

    def stop_in_the_frequency(p):
        seen.append(p.stage)
        return False if p.stage in ("assembly", "lu") else None

    with pytest.raises(hb.SolveCancelledError, match="cancel"):
        _model(i=1).solve(stop_in_the_frequency)
    first = next(i for i, s in enumerate(seen) if s in ("assembly", "lu"))
    assert "frequency_done" not in seen[first:]


def test_a_callback_does_not_change_the_results():
    # two frequency threads: the callback is called from both
    a = outputs(_small(options={"frequency_threads": 2}).solve())
    b = outputs(_small(options={"frequency_threads": 2}).solve(lambda p: None))
    assert sorted(a) == sorted(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)


def test_changes_and_queries_wait_while_the_model_calculates():
    # a change or a query from the callback would rebuild the problem the solve reads; it is
    # refused, the state and the items are not
    m = _small((1.0, 2.0))
    seen = []

    def report(done, total):
        seen.append(m.state)
        assert m.bodies["body"].mesh is not None  # the items are read meanwhile
        with pytest.raises(RuntimeError, match="the model is calculating: its inputs cannot change until"):
            m.environment.depth = 30.0
        with pytest.raises(RuntimeError, match="the model is calculating"):
            m.counts()
        with pytest.raises(RuntimeError, match="the model is calculating"):
            _ = m.results

    m.solve(report)
    assert seen == ["calculating", "calculating"] and m.state == "calculated"


def test_close_is_refused_while_the_model_calculates():
    # close() from the callback is refused; the handle and the results of the solve remain
    m = _small((1.0,))
    refused = []

    def report(done, total):
        with pytest.raises(RuntimeError, match="the model is calculating"):
            m.close()
        refused.append(True)

    r = m.solve(report)
    assert refused and m.state == "calculated" and r.added_mass is not None
    m.close()


# ------------------------------------------------------------------ the progress snapshot


def test_a_cancel_in_the_second_order():
    # cancelled from another thread once a step of the second order runs: the solve ends within a
    # second with SolveCancelledError, the model reset; the step and its items cancelled
    for step in ("potential_radiation", "potential_outer"):
        m = _second_order(panel_size=0.15)
        err = []

        def run(m=m, err=err):
            try:
                m.solve()
            except hb.SolveCancelledError as e:
                err.append(e)

        t = threading.Thread(target=run)
        t.start()
        sent = None
        while t.is_alive():
            p = m.progress()
            if sent is None and any(s.step == step and s.state == "running" for s in p.second_order):
                m.cancel()
                sent = time.perf_counter()
            time.sleep(0.005)
        t.join()
        assert sent is not None, f"{step} was not seen running"
        assert time.perf_counter() - sent < 1.0
        assert err and m.state == "reset"
        end = m.progress()
        assert end.phase == "cancelled" and all(w.state == "done" for w in end.workers)
        last = end.second_order[-1]
        assert last.step == step and last.state == "cancelled"
        assert all(i.state in ("done", "cancelled") for i in last.items)


def test_the_snapshot_from_another_thread():
    # polled from the main thread while the solve runs on another: the frequencies in the order of
    # the results, the limits apart (zero, then infinite), fractions that never decrease, the
    # running frequencies never more than the frequency threads; at the end every one done
    import threading
    import time

    # the 2048 panels of _model (a solve of seconds), two frequencies at once (checked below)
    m = model({"mesh": slow_cylinder()}, [0.0, 1.0, 1.5, 2.0, float("inf")], [0.0],
              options={"frequency_threads": 2, "inner_threads": 1})  # fmt: skip
    assert m.progress() == hb.SolveProgress(False, 0.0, 0.0, (), (), 0, 0, None, (), ())  # before the first solve
    t = threading.Thread(target=m.solve)
    t.start()
    polls = []
    while t.is_alive():
        polls.append(m.progress())
        time.sleep(0.02)
    t.join()
    end = m.progress()
    assert not end.running and end.done == end.total == 5 and end.preparation == 1.0
    assert [lp.limit for lp in end.limits] == ["zero", "infinite"]
    assert all(f.state == "done" and f.fraction == 1.0 and f.stage is None for f in end.frequencies)
    assert len(end.frequencies) == 3 and all(lp.progress.state == "done" for lp in end.limits)
    running = [p for p in polls if p.running]
    assert running and all(sum(f.state == "running" for f in p.frequencies) <= 2 for p in running)
    assert any(f.state == "running" and 0.0 < f.fraction < 1.0 for p in running for f in p.frequencies)
    for a, b in zip(polls, polls[1:], strict=False):
        for fa, fb in zip(a.frequencies, b.frequencies, strict=True):
            assert fb.fraction >= fa.fraction
        assert b.seconds >= a.seconds and b.done >= a.done


def test_the_snapshot_of_a_cancelled_solve():
    # cancelled when the limit is done (its report always arrives; a stage report of the next
    # frequency may be throttled away on a fast machine): the ordinary frequencies are cancelled
    m = _model(f=1, i=1)

    def stop(p):
        return False if p.stage == "frequency_done" and p.frequency == -1 else None

    with pytest.raises(hb.SolveCancelledError):
        m.solve(stop)
    p = m.progress()
    assert not p.running and p.done == 1 and p.total == 3
    assert [lp.progress.state for lp in p.limits] == ["done"]
    assert [f.state for f in p.frequencies] == ["cancelled", "cancelled"]
