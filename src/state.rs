use crate::loader::{load_trace_progressive, load_multi_progressive};
use crate::types::*;
#[cfg(all(target_arch = "wasm32", feature = "mt"))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

/// Fires off a trace-load job that reports its result via an mpsc channel
/// (polled every frame by `poll_loading`), mirroring the old
/// `std::thread::spawn(move || { ... })` background-job shape.
///
/// Native: `rayon::spawn` onto rayon's global pool — real worker threads
/// continuously service that pool's injector queue, so this runs
/// asynchronously in the background exactly like the `std::thread::spawn` it
/// replaces.
///
/// wasm32 + `mt`: same `rayon::spawn`, onto the wasm-bindgen-rayon Worker
/// pool `main` initialized before the event loop started (see main.rs) —
/// the load runs on a Worker and the main thread keeps rendering, so
/// `poll_loading` animates the event counter like it does natively. Gated
/// on `MT_POOL_READY` because a bare `rayon::spawn` with no live pool is
/// never picked up on wasm (confirmed empirically — nothing services the
/// global queue), which would hang the load forever; if pool init failed,
/// fall back to the synchronous inline run instead.
///
/// wasm32 without `mt`: no real threads exist at all, so run synchronously
/// inline. The job blocks the main thread for its whole duration; the
/// one-frame `pending_parse` deferral (see `Pane::run_pending_parse`) at
/// least gets a "Parsing trace..." splash painted before the freeze.
fn spawn_load_job(job: impl FnOnce() + Send + 'static) {
    #[cfg(not(target_arch = "wasm32"))]
    rayon::spawn(job);
    #[cfg(all(target_arch = "wasm32", feature = "mt"))]
    if MT_POOL_READY.load(Ordering::Relaxed) {
        rayon::spawn(job);
    } else {
        job();
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
    job();
}

/// Whether wasm-bindgen-rayon's Worker pool initialized successfully (mt
/// build only). Set by `main` after `init_thread_pool` resolves; read by
/// `spawn_load_job` to decide between a real background spawn and the
/// synchronous fallback.
#[cfg(all(target_arch = "wasm32", feature = "mt"))]
pub static MT_POOL_READY: AtomicBool = AtomicBool::new(false);

/// Single-threaded wasm build only: a load that has been announced
/// (`loading` is `Some`) but hasn't completed. There are no threads to run
/// it on, so `run_pending_parse` drives it across frames: staged for one
/// frame first (so a "Parsing trace..." splash paints before the main
/// thread does any blocking work), then stepped a time-budgeted slice of
/// the load per frame (see `SteppedParse` in loader.rs) so the splash
/// keeps animating — the event counter during the parse phase, phase text
/// otherwise — instead of freezing for the whole load.
#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
pub(crate) enum PendingParse {
    /// Inputs not yet prepared (decompress + split points).
    Single {
        name: String,
        bytes: Vec<u8>,
        counter: Arc<AtomicUsize>,
        tx: mpsc::Sender<Result<Trace, String>>,
    },
    Multi {
        rank_named_bytes: Vec<(usize, String, Vec<u8>)>,
        counter: Arc<AtomicUsize>,
        tx: mpsc::Sender<Result<Trace, String>>,
    },
    /// Prepared; stepped a time-budgeted slice per frame until `step`
    /// reports done.
    Stepping {
        plan: crate::loader::SteppedParse,
        tx: mpsc::Sender<Result<Trace, String>>,
    },
}

/// Interned name indices matching vLLM's per-generation execute wrapper span.
/// Two formats exist (see vLLM `gpu_worker.py:annotate_profile`):
///   legacy:   `execute_context_1(4)_generation_1(1)`
///   detailed: `execute_512_context_1(sq512sk388352sqsq262144sqsk198836224)_generation_0(sq0sk0sqsq0sqsk0)`
/// The detailed format has token counts in the name itself (see
/// `parse_exec_context_name`). Pulled out as a pure function so it's
/// testable without spinning up the loader pipeline, and so `poll_loading`
/// can compute it once per trace load instead of every toolbar frame.
pub(crate) fn find_exec_context_names(names: &[String]) -> Vec<usize> {
    names.iter().enumerate()
        .filter(|(_, n)| {
            n.starts_with("execute_")
                && n.contains("_context_")
                && n.contains("_generation_")
        })
        .map(|(i, _)| i)
        .collect()
}

/// Leading ASCII digits of `s`, returning the parsed number and the rest.
/// None when `s` doesn't start with a digit (or doesn't parse).
fn take_num(s: &str) -> Option<(u64, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 { return None; }
    Some((s[..end].parse().ok()?, &s[end..]))
}

/// Parse the counts out of an execute-annotation span name. Accepts both
/// the legacy `execute_context_1(4)_generation_1(1)` and detailed
/// `execute_512_context_1(sq512sk...)_generation_0(sq0sk...)` formats.
/// Returns None for names that don't match either shape.
pub(crate) fn parse_exec_context_name(name: &str) -> Option<StepBatch> {
    let rest = name.strip_prefix("execute_")?;
    // Legacy is `execute_context_1(4)...` — after the prefix strip the
    // marker sits at the string start (no total-tokens prefix). Detailed is
    // `execute_512_context_1(...)` — marker after a total-tokens prefix.
    let (total_tokens, rest) = if let Some(r) = rest.strip_prefix("context_") {
        (None, r)
    } else {
        let (tot, r) = rest.split_once("_context_")?;
        (Some(tot.parse().ok()?), r)
    };
    // rest: "<n>(<ctx inner>)_generation_<n>(<gen inner>)
    let (ctx_seg, gen_seg) = rest.split_once(")_generation_")?;
    let (ctx_n_str, ctx_inner) = ctx_seg.split_once('(')?;
    let ctx_n: u64 = ctx_n_str.parse().ok()?;
    let (gen_n_str, gen_inner) = gen_seg.split_once('(')?;
    let gen_n: u64 = gen_n_str.parse().ok()?;
    let gen_inner = gen_inner.strip_suffix(')')?;

    // Inner is "<tokens>" (legacy) or
    // "sq<tokens>sk<kv>sqsq<qq>sqsk<qk>" (detailed). Parse positionally —
    // the "sqsq"/"sqsk" markers contain "sk", so a plain split("sk") would
    // mangle the KV number. The two trailing compute proxies (qq/qk) aren't
    // needed for the summary table.
    let parse_inner = |inner: &str| -> Option<(u64, Option<u64>)> {
        if let Some(r) = inner.strip_prefix("sq") {
            let (tokens, r) = take_num(r)?;
            let kv = r.strip_prefix("sk").and_then(take_num).map(|(kv, _)| kv);
            Some((tokens, kv))
        } else {
            Some((inner.parse().ok()?, None))
        }
    };
    let (ctx_tokens, ctx_kv) = parse_inner(ctx_inner)?;
    let (gen_tokens, gen_kv) = parse_inner(gen_inner)?;

    let mut b = StepBatch {
        total_tokens: total_tokens.unwrap_or(0),
        n_ctx: ctx_n, n_gen: gen_n, ctx_tokens, ctx_kv, gen_tokens, gen_kv,
    };
    if b.total_tokens == 0 {
        b.total_tokens = b.ctx_tokens + b.gen_tokens;
    }
    Some(b)
}

/// Derive the vLLM scheduler-step sequence from the trace's
/// `execute_context_*` spans (see `StepInfo`). Returns an empty Vec when the
/// trace has no such spans (non-vLLM trace, or none on any loaded track).
///
/// Pure function of the `Trace` so it's unit-testable and recomputable after
/// clock sync / reload shifts timestamps.
pub(crate) fn compute_steps(trace: &Trace, exec_context_names: &[usize]) -> Vec<StepInfo> {
    if exec_context_names.is_empty() { return Vec::new(); }
    let name_set: std::collections::HashSet<u32> =
        exec_context_names.iter().map(|&i| i as u32).collect();

    // Collect (ts, end, track_idx, name_idx) for every exec-context span,
    // preferring the GPU-side projection (`gpu_user_annotation`): it
    // brackets the actual GPU work rather than the CPU launch loop, and
    // appears on every CUDA stream so per-rank assignment is direct. Fall
    // back to CPU-side `user_annotation` for traces where the GPU
    // projection was dropped.
    let gpu_side: Vec<(f64, f64, usize, u32)> = collect_exec_spans(trace, &name_set, "gpu_user_annotation");
    let spans = if gpu_side.is_empty() {
        collect_exec_spans(trace, &name_set, "user_annotation")
    } else {
        gpu_side
    };
    if spans.is_empty() { return Vec::new(); }

    // Reference track: the one with the most spans. Its spans are sequential
    // and non-overlapping (one bracket per step per stream), so they define
    // the canonical step list — no gap-based clustering needed, which would
    // be fragile across skewed clocks.
    //
    // Ties are broken by total span duration, then by track index. The
    // tie-breaks are load-bearing: `gpu_user_annotation` projects onto
    // EVERY CUDA stream, so on a real multi-stream trace dozens of tracks
    // tie at exactly one span per step, and which one wins decides how
    // every other span is assigned. The old `max_by_key` over a HashMap
    // resolved those ties by hash iteration order — a per-call lottery
    // where the same trace computed different step bounds on every
    // recompute, and a partial stream (one whose brackets cover only the
    // head or tail of each step) as reference leaves spans overlapping no
    // interval, which the nearest-interval fallback then attaches to the
    // wrong step, stretching bounds across neighboring steps' kernels.
    // Total duration prefers a stream that brackets the whole step — a
    // projection can't exceed its CPU annotation's window, so the longest
    // per-step brackets are the ones closest to each step's true GPU
    // extent — and the index keeps the pick stable when several streams
    // are equally good. Track indices are unique, so the composite order
    // is total and the max is the same regardless of iteration order.
    let mut per_track: std::collections::HashMap<usize, (usize, f64)> = Default::default();
    for &(s0, s1, ti, _) in &spans {
        let e = per_track.entry(ti).or_default();
        e.0 += 1;
        e.1 += s1 - s0;
    }
    let ref_track = per_track.into_iter()
        .max_by(|a, b| {
            a.1.0.cmp(&b.1.0)
                .then(a.1.1.total_cmp(&b.1.1))
                .then(b.0.cmp(&a.0))
        })
        .unwrap().0;
    let mut ref_spans: Vec<(f64, f64, u32)> = spans.iter()
        .filter(|&&(_, _, ti, _)| ti == ref_track)
        .map(|&(t0, t1, _, name)| (t0, t1, name))
        .collect();
    ref_spans.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign every span to the reference interval it overlaps most. The
    // spans are ts-sorted per track by the loader, but the combined list
    // isn't globally sorted, and overlap-most needs interval arithmetic per
    // span anyway — a linear scan over reference intervals (advanced by a
    // moving cursor, since a span can't overlap two far-apart steps) is
    // O(spans + steps).
    let n = ref_spans.len();
    let mut bounds = vec![(f64::MAX, f64::MIN); n]; // (min t0, max t1)
    let mut rank_sets: Vec<std::collections::HashSet<usize>> = (0..n).map(|_| Default::default()).collect();
    // Per (step, rank) assigned intervals, for the union-based gpu_dur below.
    let mut rank_spans: Vec<std::collections::HashMap<usize, Vec<(f64, f64)>>> =
        (0..n).map(|_| Default::default()).collect();
    // Rank -> DP-group id (from `rank_paths` filenames, when the trace is a
    // multi-rank merge). Ranks in the same DP group share one scheduler and
    // annotate identical batch info, so each group's parsed batch counts
    // once per step; different groups run independent schedulers whose local
    // batches are summed. See `step_dp_groups`.
    let rank_dp = step_dp_groups(trace);
    // One parsed batch per distinct DP group per step, for aggregation.
    let mut group_batches: Vec<Vec<(u32, Option<StepBatch>)>> = (0..n).map(|_| Vec::new()).collect();
    for &(s0, s1, ti, name) in &spans {
        let track = &trace.tracks[ti];
        let rank = parse_rank(&track.label);
        // Best-overlap reference interval. Reference spans are disjoint, so
        // at most two can overlap a given span; scan a window around where
        // the span falls.
        let mut best = (0usize, 0f64);
        for (i, &(r0, r1, _)) in ref_spans.iter().enumerate() {
            if r0 >= s1 { break; }
            let ov = (s1.min(r1) - s0.max(r0)).max(0.0);
            if ov > best.1 { best = (i, ov); }
        }
        if best.1 <= 0.0 {
            // Span falls between reference intervals (skewed rank, or an
            // exec-context the reference stream didn't see). Attach it to the
            // nearest interval so its time still counts.
            let nearest = ref_spans.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.0 - s0).abs() + (a.1 - s1).abs();
                    let db = (b.0 - s0).abs() + (b.1 - s1).abs();
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(i, _)| i).unwrap_or(0);
            best.0 = nearest;
        }
        let i = best.0;
        // Batch bookkeeping: ranks in the same DP group share one scheduler
        // and annotate identical batch info, so each group contributes its
        // parsed batch once per step (first span wins); different groups run
        // independent schedulers whose local batches are summed at the end.
        let dp = rank.and_then(|r| rank_dp.get(&r).copied()).flatten();
        if let Some(dp) = dp {
            let gb = &mut group_batches[i];
            if !gb.iter().any(|&(d, _)| d == dp) {
                gb.push((dp, parse_exec_context_name(&trace.names[name as usize])));
            }
        }
        bounds[i].0 = bounds[i].0.min(s0);
        bounds[i].1 = bounds[i].1.max(s1);
        rank_sets[i].insert(rank.unwrap_or(ti));
        rank_spans[i].entry(rank.unwrap_or(ti)).or_default().push((s0, s1));
    }

    // Kernel counts: for each step, binary-search each GPU track's ts-sorted
    // events for [t0, t1) and count `cat == "kernel"`. Load-time only, so
    // O(steps * gpu_tracks * log n) is fine (30 steps * 256 tracks here).
    let kernel_cat: Vec<u32> = trace.cats.iter()
        .enumerate().filter(|(_, c)| c.as_str() == "kernel").map(|(i, _)| i as u32).collect();
    let gpu_tracks: Vec<usize> = trace.tracks.iter().enumerate()
        .filter(|(_, t)| t.gpu).map(|(ti, _)| ti).collect();

    let mut steps = Vec::with_capacity(n);
    let mut prev_t1 = 0.0;
    for i in 0..n {
        let (t0, t1) = bounds[i];
        if t1 <= t0 { continue; } // reference interval with no surviving span (shouldn't happen)
        // GPU busy: per rank, the UNION of that rank's assigned spans —
        // the projections of one annotation onto parallel streams overlap
        // in time, so summing raw span durations counts the same GPU time
        // once per stream (on a ~50-stream trace the old sum/n_ranks
        // reported ~3.5x the step's wall time). Then the median across
        // ranks: the typical rank's busy time, robust against one
        // straggler rank stretching the step.
        let mut rank_durs: Vec<f64> = rank_spans[i].values_mut()
            .map(|v| union_len(v)).collect();
        let gpu_dur = if rank_durs.is_empty() { 0.0 } else { crate::types::median_inplace(&mut rank_durs) };
        let mut kernel_count = 0u32;
        if !kernel_cat.is_empty() {
            for &ti in &gpu_tracks {
                let evs = &trace.tracks[ti].events;
                let lo = evs.partition_point(|e| e.ts < t0);
                let mut k = lo;
                while k < evs.len() && evs[k].ts < t1 {
                    if kernel_cat.contains(&evs[k].cat) { kernel_count += 1; }
                    k += 1;
                }
            }
        }
        let idle_before = if steps.is_empty() { 0.0 } else { (t0 - prev_t1).max(0.0) };
        prev_t1 = t1;
        // Batch composition. With DP info, the total is the sum over the
        // per-group local batches (kept for display); without it
        // (single-rank trace, or a merge without parseable filenames) the
        // reference span's own parsed batch is already the whole batch.
        let gbs = &group_batches[i];
        let (batch, group_batches) = if gbs.is_empty() {
            (parse_exec_context_name(&trace.names[ref_spans[i].2 as usize]), Vec::new())
        } else {
            let mut acc: Option<StepBatch> = None;
            let mut kept: Vec<(u32, StepBatch)> = Vec::new();
            for &(dp, b) in gbs {
                if let Some(b) = b {
                    acc = Some(match acc { Some(a) => a.sum(&b), None => b });
                    kept.push((dp, b));
                }
            }
            (acc, kept)
        };
        steps.push(StepInfo {
            t0, t1, gpu_dur, idle_before,
            n_ranks: rank_sets[i].len() as u32,
            kernel_count,
            batch,
            group_batches,
        });
    }
    steps
}

/// Total length covered by the given intervals (sort + merge sweep).
/// Reorders `spans`; callers pass a scratch list. Used by `compute_steps`
/// to fold one rank's parallel-stream projections into that rank's actual
/// busy time for a step.
fn union_len(spans: &mut [(f64, f64)]) -> f64 {
    spans.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut total = 0.0f64;
    let mut cur_end = f64::NEG_INFINITY;
    for &(s0, s1) in spans.iter() {
        if s1 > cur_end {
            total += s1 - s0.max(cur_end);
            cur_end = s1;
        }
    }
    total
}

/// Whether a cat counts as GPU work: `kernel` plus `gpu_*` EXCEPT
/// `gpu_user_annotation` (annotations are CPU-side brackets projected onto
/// every stream — counting them as busy would erase exactly the idle the
/// Gaps tab and the selection's empty-space summary exist to measure).
/// Shared by `compute_gaps_in` and `selection_gpu_summary`.
pub(crate) fn is_gpu_work_cat(c: &str) -> bool {
    c == "kernel" || (c.starts_with("gpu_") && c != "gpu_user_annotation")
}

/// GPU idle gaps ("bubbles") for the Gaps tab: per rank scope, the wall-time
/// intervals where NO gpu-work event covers that rank's timeline, biggest
/// first, capped at `GAP_TOP_N` and annotated with what the rank's CPU
/// threads were doing inside each one.
///
/// "GPU work" is cat `kernel` plus `gpu_*` EXCEPT `gpu_user_annotation`:
/// annotations are CPU-side brackets projected onto every stream, so they
/// span each step's internal idle — counting them as busy would erase
/// exactly the gaps this exists to find. Zero-duration events are skipped
/// for the same reason (an instant adds no busy time but would split a gap
/// in two). Scoping is per rank (`parse_rank` of the track label) so a gap
/// on one rank while another works still counts; unlabeled tracks (a
/// single-rank trace) share one scope.
///
/// "CPU busy" counts only cats kineto instruments as executed host work:
/// `cpu_op` (aten ops) plus the CUDA API cats (`runtime`/`cuda_runtime`,
/// `cuda_driver`). `python_function` and `user_annotation` spans are stack
/// extent, not work — a thread parked in `queue.get` for 7 seconds holds an
/// open python frame the whole time, and counting that as busy would label
/// every starved gap "host-bound" (found on a real with_stack trace). Those
/// cats still show in the timeline itself for context; they just don't
/// feed the fractions or the top-ops list.
///
/// Pure function of the `Trace` so it's unit-testable and recomputable
/// after clock sync / reload shifts timestamps.
///
/// `compute_gaps_in` additionally applies a `GapFilter`: window clip, max
/// length, and rank scoping, all BEFORE the top-N cut, so profiler-artifact
/// gaps don't consume list slots (see `GapFilter`'s doc).
pub(crate) fn compute_gaps_in(trace: &Trace, f: GapFilter) -> Vec<GapInfo> {
    let mut is_work = BitSet::with_len(trace.cats.len());
    let mut any_work = false;
    for (i, c) in trace.cats.iter().enumerate() {
        if is_gpu_work_cat(c) {
            is_work.set(i, true);
            any_work = true;
        }
    }
    if !any_work { return Vec::new(); }
    // Host-work cats for the CPU annotation (see the doc above): what the
    // thread actually executed, vs. the stack extent python/annotation
    // frames describe. `cuda_runtime` is the cat name real kineto traces
    // carry ("runtime" is the older spelling; driver-level calls belong to
    // the same launch path).
    let is_cpu_work = |c: &str| {
        c == "cpu_op" || c == "runtime" || c == "cuda_runtime" || c == "cuda_driver"
    };
    let is_runtime = |c: &str| c == "runtime" || c == "cuda_runtime" || c == "cuda_driver";

    // Per-scope busy intervals from GPU tracks, plus one focus track and the
    // CPU track list per scope for the annotation pass.
    let mut scopes: std::collections::HashMap<usize, Vec<(f64, f64, u32)>> = Default::default();
    let mut focus_track: std::collections::HashMap<usize, u32> = Default::default();
    let mut cpu_tracks: std::collections::HashMap<usize, Vec<usize>> = Default::default();
    for (ti, track) in trace.tracks.iter().enumerate() {
        let scope = parse_rank(&track.label).unwrap_or(usize::MAX);
        if track.gpu {
            focus_track.entry(scope).or_insert(ti as u32);
            for ev in &track.events {
                if ev.dur > 0.0 && is_work[ev.cat as usize] {
                    scopes.entry(scope).or_default().push((ev.ts, ev.ts + ev.dur, ev.name));
                }
            }
        } else {
            cpu_tracks.entry(scope).or_default().push(ti);
        }
    }

    // Raw gaps across all scopes: (scope, t0, t1, prev, next, focus_track).
    let mut raw: Vec<(usize, f64, f64, Option<u32>, Option<u32>, u32)> = Vec::new();
    for (scope, mut ivs) in scopes {
        // Selection scoping: ranks the selection rect didn't cross are
        // skipped entirely (an empty scope list therefore yields no gaps).
        if f.scopes.as_ref().is_some_and(|s| !s.contains(&scope)) { continue; }
        ivs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
        // Merge into the union of busy intervals. Each merged interval
        // remembers the name that opens it (earliest start) and closes it
        // (latest end) — the gap's "after"/"resumes" kernels.
        let mut merged: Vec<(f64, f64, u32, u32)> = Vec::new();
        for (s, e, name) in ivs {
            match merged.last_mut() {
                Some(m) if s <= m.1 => {
                    if e > m.1 { m.1 = e; m.3 = name; }
                }
                _ => merged.push((s, e, name, name)),
            }
        }
        let ft = focus_track[&scope];
        for w in merged.windows(2) {
            let len = w[1].0 - w[0].1;
            if len >= crate::types::MIN_GAP_US {
                raw.push((scope, w[0].1, w[1].0, Some(w[0].3), Some(w[1].2), ft));
            }
        }
    }
    // User filter before the sort/top-N cut: clip each gap to the window
    // and drop edge-clipped shavings, so artifact-sized gaps outside the
    // viewed region don't take list slots from real bubbles.
    if f.window.is_some() {
        raw.retain_mut(|g| {
            if let Some((w0, w1)) = f.window {
                g.1 = g.1.max(w0);
                g.2 = g.2.min(w1);
            }
            g.2 - g.1 >= crate::types::MIN_GAP_US
        });
    }
    raw.sort_unstable_by(|a, b| (b.2 - b.1).partial_cmp(&(a.2 - a.1)).unwrap());
    raw.truncate(crate::types::GAP_TOP_N);

    // CPU annotation, only for the kept top gaps.
    let mut gaps: Vec<GapInfo> = Vec::with_capacity(raw.len());
    for (scope, g0, g1, prev_name, next_name, ft) in raw {
        let dur = g1 - g0;
        let cts = cpu_tracks.get(&scope).map(|v| &v[..]).unwrap_or(&[]);
        let mut any: Vec<(f64, f64)> = Vec::new();
        let mut runtime: Vec<(f64, f64)> = Vec::new();
        let mut per_name: std::collections::HashMap<u32, (u32, Vec<(f64, f64)>)> = Default::default();
        for &ti in cts {
            let track = &trace.tracks[ti];
            let start = bisect_overlap(&track.events, &track.prefix_max_dur, g0);
            let end = track.events.partition_point(|e| e.ts <= g1);
            for ev in &track.events[start..end] {
                // Only executed host work (cpu_op / CUDA API) — python and
                // annotation frames describe where the stack was parked, not
                // that the thread was running (see the fn doc).
                let cat = trace.cats[ev.cat as usize].as_str();
                if !is_cpu_work(cat) { continue; }
                // Clip to the gap; skip events that only touch its edges.
                let s = ev.ts.max(g0);
                let e = (ev.ts + ev.dur).min(g1);
                if e <= s { continue; }
                any.push((s, e));
                if is_runtime(cat) { runtime.push((s, e)); }
                let en = per_name.entry(ev.name).or_insert((0, Vec::new()));
                en.0 += 1;
                en.1.push((s, e));
            }
        }
        let mut cpu_top: Vec<(u32, u32, f64)> = per_name.into_iter()
            .map(|(name, (count, mut spans))| (name, count, union_len(&mut spans)))
            .collect();
        cpu_top.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        cpu_top.truncate(3);
        gaps.push(GapInfo {
            rank: if scope == usize::MAX { None } else { Some(scope) },
            t0: g0,
            t1: g1,
            cpu_busy_frac: union_len(&mut any) / dur,
            runtime_frac: union_len(&mut runtime) / dur,
            has_cpu: !cts.is_empty(),
            cpu_top,
            prev_name,
            next_name,
            focus_track: ft,
        });
    }
    gaps
}

/// The selection's GPU empty space, PER RANK, for the Selection tab
/// summary: the selection's GPU-work events (from the frozen
/// `finished_sel_events` set), clipped to its time range, grouped by rank
/// scope (`parse_rank` of the track label; unlabeled single-rank traces
/// share one scope). Per rank: span from that rank's leftmost to rightmost
/// kernel, `busy` the union of its clipped spans — concurrent streams of
/// the same rank must not double-count, and one rank working says nothing
/// about another's idle. Entries sorted by rank; empty when the selection
/// holds no GPU work events (CPU-only selection, or annotations only).
pub(crate) fn selection_gpu_summary(
    trace: &Trace,
    events: &std::collections::HashSet<(u32, u32)>,
    sel: [f64; 4],
) -> Vec<SelGpuSummary> {
    // Rect layout is [t_a, t_b, y_a, y_b] (see where the drag builds it in
    // draw_timeline) — same normalization capture_sel_events applies. Reading
    // t_b from index 2 instead mixed a y pixel into the time window and
    // clipped away every selected event (the summary never fired).
    let (s0, s1) = if sel[0] <= sel[1] { (sel[0], sel[1]) } else { (sel[1], sel[0]) };
    let mut per_rank: Vec<(Option<usize>, Vec<(f64, f64)>)> = Vec::new();
    for &(ti, ei) in events {
        let track = &trace.tracks[ti as usize];
        if !track.gpu { continue; }
        let ev = &track.events[ei as usize];
        if ev.dur <= 0.0 || !is_gpu_work_cat(trace.cats[ev.cat as usize].as_str()) { continue; }
        let a = ev.ts.max(s0);
        let b = (ev.ts + ev.dur).min(s1);
        if b <= a { continue; }
        let rank = parse_rank(&track.label);
        match per_rank.iter_mut().find(|(r, _)| *r == rank) {
            Some((_, spans)) => spans.push((a, b)),
            None => per_rank.push((rank, vec![(a, b)])),
        }
    }
    let mut out: Vec<SelGpuSummary> = per_rank.into_iter()
        .map(|(rank, mut spans)| SelGpuSummary {
            rank,
            t0: spans.iter().map(|s| s.0).fold(f64::INFINITY, f64::min),
            t1: spans.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max),
            busy: union_len(&mut spans),
        })
        .collect();
    out.sort_by_key(|g| g.rank);
    out
}

/// Maps rank id -> DP-group id for a merged multi-rank trace, from the
/// per-rank source filenames persisted in `Trace::rank_paths`
/// (`dp{N}_pp{P}_tp{T}_dcp{D}_ep{E}_rank{R}...`, built by vLLM's
/// `get_custom_suffix` in distributed/utils.py).
///
/// Why grouping by DP matters: with external data parallelism, every DP
/// group runs its own scheduler and its workers' execute-annotations
/// (`gpu_worker.py:annotate_profile`) describe only that group's local
/// batch. Ranks in the same group (all TP/EP/DCP/PP members — they share
/// the group's scheduler output) annotate identically, so per-group batches
/// must be deduplicated, and different groups' local batches summed. This
/// is topology-agnostic: whatever dp/tp/ep/pp/dcp shape the job ran, the
/// filename's `dp` index is the scheduler boundary, and we never assume a
/// group size.
///
/// Ranks with no parseable `dp` index map to `None` (each such rank gets a
/// singleton group, matching `sync_multi_rank_clocks`' solo fallback), and
/// empty `rank_paths` (single-rank trace, or a merge whose sources didn't
/// carry the suffix) yields an empty map — the caller then just uses the
/// reference span's batch, which is already the whole batch for a
/// single-scheduler trace.
fn step_dp_groups(trace: &Trace) -> std::collections::HashMap<usize, Option<u32>> {
    let mut map = std::collections::HashMap::new();
    for (rank, path) in &trace.rank_paths {
        let fname = std::path::Path::new(path)
            .file_name().and_then(|s| s.to_str()).unwrap_or(path.as_str());
        map.insert(*rank, crate::loader::parse_dp_tp(fname).0);
    }
    map
}

/// All spans on any track whose name matches `name_set` and whose category
/// is exactly `cat`, as (start, end, track_idx, name_idx). Track event lists
/// are ts-sorted per track by the loader; the combined output is NOT
/// globally sorted (callers don't need it to be).
fn collect_exec_spans(
    trace: &Trace,
    name_set: &std::collections::HashSet<u32>,
    cat: &str,
) -> Vec<(f64, f64, usize, u32)> {
    let cat_ids: Vec<u32> = trace.cats.iter().enumerate()
        .filter(|(_, c)| c.as_str() == cat).map(|(i, _)| i as u32).collect();
    if cat_ids.is_empty() { return Vec::new(); }
    let mut out = Vec::new();
    for (ti, track) in trace.tracks.iter().enumerate() {
        for ev in &track.events {
            if name_set.contains(&ev.name) && cat_ids.contains(&ev.cat) {
                out.push((ev.ts, ev.ts + ev.dur, ti, ev.name));
            }
        }
    }
    out
}

pub(crate) fn parse_rank(label: &str) -> Option<usize> {
    let l = label.trim_start();
    if l.starts_with("Rank ") {
        l[5..].split_once(' ').and_then(|(n, _)| n.parse().ok())
    } else {
        None
    }
}

/// Default view-level track order: a stable sort by numeric rank (via
/// `parse_rank`), not by track index or label text. Deliberately computed
/// here rather than relying on `merge_traces` writing tracks in the right
/// order to begin with — this way the *view* is always correct regardless
/// of how the underlying `Trace` was produced, including a `.tvcache`
/// export that was itself built and saved before a rank-ordering fix
/// existed (the label text it stored still has the numeric rank in it;
/// only the on-disk *track order* predates the fix). A trace with no
/// "Rank N" labels at all (not a multi-rank merge) gets the same `None`
/// key for every track, so the stable sort leaves it in its original order.
pub(crate) fn default_track_order(tracks: &[Track]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..tracks.len()).collect();
    order.sort_by_key(|&i| parse_rank(&tracks[i].label).unwrap_or(usize::MAX));
    order
}

pub struct Pane {
    pub trace: Option<Trace>,
    pub view: View,
    pub loading: Option<mpsc::Receiver<Result<Trace, String>>>,
    pub error: Option<String>,
    pub trace_path: String,
    pub show_cpu: bool,
    pub merge_gpu: bool,
    pub selected: Option<EventRef>,
    pub multi_select_name: Option<u32>,
    pub search: String,
    pub search_focus: bool,
    pub select_all_pending: bool,
    pub search_mask: BitSet,
    pub search_nav: Vec<(f64, u32, u32)>,
    pub search_cursor: usize,
    pub prev_search: String,
    pub selection: Option<[f64; 4]>,
    pub finished_sel: Option<[f64; 4]>,
    /// The concrete `(track_idx, event_idx)` set `finished_sel` refers to,
    /// resolved once when the drag finishes (see `capture_sel_events`). Every
    /// consumer of `finished_sel` after that point (stats, extract, copy, the
    /// rendered highlight) reads this frozen set instead of re-deriving track
    /// membership from `finished_sel`'s raw Y range against whatever the
    /// CURRENT layout happens to be — otherwise toggling Show CPU, reordering
    /// tracks, or resizing the bottom panel (which can change track heights)
    /// silently changes which events are "selected".
    pub finished_sel_events: std::collections::HashSet<(u32, u32)>,
    pub selection_stats: Vec<SelectionEntry>,
    /// GPU empty space inside the finished selection, per rank (see
    /// `selection_gpu_summary`): per rank, span from its leftmost to
    /// rightmost kernel and the busy union within. Empty = no GPU work in
    /// the selection. Recomputed with `selection_stats` in
    /// `rebuild_selection_stats`.
    pub sel_gpu: Vec<SelGpuSummary>,
    pub selection_dirty: bool,
    pub collapsed: BitSet,
    pub track_scales: Vec<f32>,
    pub even_spacing: bool,
    pub geom: PaneGeom,
    pub hidden_names: BitSet,
    /// Cache key for `merged_gpu_groups` below. The pack is VIEW-INDEPENDENT
    /// (whole-span packing, see `MergedPack`), so the key is only
    /// (hidden_names, track_order) — panning and zooming must NOT rebuild
    /// it: a window-relative pack is what made events straddling the left
    /// edge visibly re-lane frame after frame during a pan (off-screen
    /// influencers dropping out of the collect window) and cost a full
    /// repack on every view-change frame.
    pub merge_cache_key: Option<(BitSet, Vec<usize>)>,
    /// The merged multi-rank view's per-rank-group row data (packed events +
    /// max depth), validated against `merge_cache_key` above. This must live
    /// here, not on the shared `DrawBuf` (where it lived before it was
    /// cached) — once frame-to-frame reuse was introduced, a pane whose own
    /// `merge_cache_key` still matched could otherwise silently read back
    /// whatever a *different* pane's render had just overwritten this with,
    /// panicking on (track_idx, event_idx) pairs that don't exist in this
    /// pane's own trace. Harmless before caching, since it was always fully
    /// rebuilt within the same frame it was read, regardless of which pane's
    /// turn it was.
    pub merged_gpu_groups: Vec<MergedGpuGroup>,
    /// Interned name indices matching "execute_context" — vLLM's per-generation
    /// wrapper span. Computed once per trace load (see `poll_loading`), not
    /// per toolbar frame, since it only depends on `trace.names`.
    pub exec_context_names: Vec<usize>,
    /// vLLM scheduler steps derived from those spans (see `compute_steps`).
    /// Empty for traces without execute_context spans. Recomputed on load,
    /// after clock sync (which shifts timestamps), and after auto-reload.
    pub steps: Vec<StepInfo>,
    /// Current step for `[`/`]` navigation and ruler highlighting. Index into
    /// `steps`; meaningless while `steps` is empty.
    pub step_cursor: usize,
    /// GPU idle gaps for the Gaps tab (see `compute_gaps`). `None` = dirty:
    /// computed lazily by `ensure_gaps` the first time the tab opens (the
    /// sweep sorts every GPU event — pointless on loads where the user never
    /// looks), cleared on reload and clock sync, which shift the timestamps
    /// the gaps are derived from.
    pub gaps: Option<Vec<GapInfo>>,
    /// Gaps-tab row the user last clicked: highlighted in the table and
    /// drawn as a red band on the timeline/ruler. Index into `gaps`, cleared
    /// wherever `gaps` is invalidated (the indices go stale with it).
    pub focused_gap: Option<usize>,
    /// Gaps-tab detection filter (window; see `GapFilter`).
    /// Changing it in the UI clears `gaps`/`focused_gap` to force a
    /// recompute. The window is timestamp-based, so it's also cleared on
    /// reload and clock sync like `gaps` itself.
    pub gap_filter: GapFilter,
    /// Rank scopes the finished selection's y-range crossed, frozen at
    /// drag-finish time (see `capture_sel_scopes`). While a non-zero-width
    /// selection is active, `ensure_gaps` scopes gap detection to these —
    /// empty when nothing is selected.
    pub sel_scopes: Vec<usize>,
    pub pending_tab: Option<BottomTab>,
    /// Which bottom-panel tab is currently active, mirrored from the imgui
    /// tab bar each frame (the TabItem body that runs is the active one).
    /// Lets event handlers decide whether to switch tabs — e.g. a finished
    /// selection switches away from Detail (which has nothing to show for
    /// it) but never steals focus from Gaps/Steps.
    pub bottom_tab: BottomTab,
    /// Single-threaded wasm build only: a load whose parse hasn't run yet.
    /// `open_from_bytes`/`open_multi_from_bytes` stage the job here instead
    /// of running it (no threads exist to background it with); `run_pending_parse`
    /// executes it a frame later. See `PendingParse` for why the delay.
    #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
    pub(crate) pending_parse: Option<PendingParse>,
    /// Whether a frame has rendered since `pending_parse` was staged. The
    /// staging frame only announces (splash shows "Parsing trace...");
    /// the frame after that runs the parse, once the announcement painted.
    #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
    pending_parse_painted: bool,
    /// Track index whose row draw_timeline should scroll into view as part of an
    /// in-flight search zoom. Consumed (cleared) on the next timeline draw.
    pub pending_focus: Option<u32>,
    pub sort_col: usize,
    pub sort_asc: bool,
    /// (generation, sort_col, sort_asc, row count, is_individual_mode,
    /// overlap_aware) from the last time `sort_idx` was actually recomputed
    /// for THIS pane. Owned per-pane, not on the shared `DrawBuf` — only one
    /// pane renders per frame (see the tab strip in main.rs), so a shared
    /// cache would compare against whichever *other* pane last rendered and
    /// could spuriously "match" and show that pane's stale rows instead of
    /// recomputing.
    pub sort_cache_key: Option<(u64, usize, bool, usize, bool, bool)>,
    pub sort_idx: Vec<usize>,
    /// (event name, show_cpu, bucket count) the Detail tab's
    /// duration-distribution histogram was last computed for, and the
    /// cached bucket counts — same per-pane-ownership reasoning as
    /// `sort_cache_key` (a name id is only unique within this pane's own
    /// trace, so two different open traces routinely reuse the same low
    /// ids). Bucket count is part of the key because it now scales with the
    /// panel's width (see `DETAIL_HIST_TARGET_BAR_W`), so a resize needs a
    /// rebucket too, not just a name/visibility change.
    pub detail_hist_key: Option<(u32, bool, usize)>,
    pub detail_hist_bins: Vec<u32>,
    pub detail_hist_min: f64,
    pub detail_hist_max: f64,
    pub detail_hist_mean: f64,
    pub detail_hist_median: f64,
    pub sel_aggregate: bool,
    pub label_w: f32,
    pub sel_median: f64,
    pub sel_agg_stats: Vec<KernelStats>,
    /// Earliest start time of each aggregate row's events (same row order as
    /// `sel_agg_stats`), for the Selection table's "Start" column — sorting
    /// it orders the kernels chronologically by first occurrence. The
    /// per-event table doesn't need a parallel array: each row's (track,
    /// event) ref is enough to read its `ts` directly in `draw_stats_table`.
    pub sel_agg_starts: Vec<f64>,
    pub sel_individual: Vec<KernelStats>,
    /// (track_idx, event_idx) parallel to `sel_individual`, so the stats table
    /// can look up each row's raw args (e.g. CUDA occupancy-limiting factor).
    pub sel_individual_refs: Vec<(u32, u32)>,
    /// "Overlap aware" toggle for the Selection tab's aggregate table. When
    /// set, each row's Total is that kernel's *shared* time — every instant
    /// of wall time split equally among the selected kernels running it,
    /// per rank scope — instead of the plain sum of durations. Concurrent
    /// launches of the kernel itself count once, and time overlapped by
    /// other kernels is discounted to its share. Off by default; only
    /// meaningful in aggregate mode (a per-event row is a single launch).
    pub sel_overlap: bool,
    /// Shared-time totals parallel to `sel_agg_stats` (same row order),
    /// computed lazily by `ensure_sel_overlap_totals` the first frame the
    /// overlap mode is shown after a selection rebuild. Lazy because the
    /// sweep sorts every selected event's interval endpoints — comparable
    /// to the whole of `compute_aggregates` — and pointless on rebuilds for
    /// selections where the user never toggles the mode.
    pub sel_overlap_totals: Vec<f64>,
    /// `sel_generation` the cached `sel_overlap_totals` were computed for.
    pub sel_overlap_gen: u64,
    /// Bumped every time `compute_aggregates` rebuilds `sel_agg_stats` /
    /// `sel_individual`. `draw_stats_table` uses this to skip re-sorting on
    /// redraws where the selection hasn't actually changed — the table
    /// otherwise re-sorted unconditionally on every redraw (which fires on
    /// every mouse-move event, not just clicks), measured at ~120ms for a
    /// 1M-row selection even on a plain numeric column.
    pub sel_generation: u64,
    pub track_order: Vec<usize>,
    pub auto_reload: bool,
    pub reload_paths: Vec<(usize, String)>,
    pub reload_dir: Option<String>,
    pub cache_dir: Option<String>,
    pub loading_events: Arc<AtomicUsize>,
    /// Result of the last "Export GPU" click (success message, or an
    /// error) — shown next to the button until the next click replaces it.
    #[cfg(not(target_arch = "wasm32"))]
    pub export_message: Option<(bool, String)>,
    /// Background `gh`/`git`-based gist upload started by "Share via Gist"
    /// (see `start_share_upload`) — `None` once no upload is in flight.
    #[cfg(not(target_arch = "wasm32"))]
    pub share_link_job: Option<mpsc::Receiver<Result<String, String>>>,
    /// Result of the last "Share via Gist" click (the finished `?gist=`
    /// link, or an error) — shown next to the button until replaced.
    #[cfg(not(target_arch = "wasm32"))]
    pub share_link_result: Option<(bool, String)>,
    /// "Include CPU data" checkbox next to "Share via Gist" — persisted
    /// across frames like any other UI toggle. When set, uploads the whole
    /// trace (`export_full_web`) instead of the GPU-only, args-stripped
    /// default (`export_gpu_only_web`).
    #[cfg(not(target_arch = "wasm32"))]
    pub share_include_cpu: bool,
    /// Result of the last "Sync Clocks" click (success summary, or an
    /// error) — shown next to the button until the next click replaces it.
    pub sync_message: Option<(bool, String)>,
    /// View > Color > By Duration: heat-map events by duration instead of
    /// hashing the name. Per-pane like show_cpu/merge_gpu so two tabs can
    /// use different modes.
    pub color_by_dur: bool,
    /// Log10 duration bounds (p5, p95) the heat map maps onto blue→red,
    /// computed lazily by `ensure_heat_bounds` the first frame the mode is
    /// on, and cleared whenever a new trace lands. Percentiles (sampled to
    /// ~100k events) rather than min/max so one huge memcpy outlier can't
    /// push every kernel into the cold end of the scale. GPU-track events
    /// only: CPU spans include whole-step wrappers that would otherwise
    /// dominate the top of the range and flatten the kernel scale.
    pub heat_bounds: Option<(f64, f64)>,
}

impl Pane {
    pub fn new() -> Self {
        Self {
            trace: None,
            view: View::default(),
            loading: None,
            error: None,
            trace_path: String::new(),
            show_cpu: false,
            merge_gpu: true,
            selected: None,
            multi_select_name: None,
            search: String::with_capacity(256),
            search_focus: false,
            select_all_pending: false,
            search_mask: BitSet::new(),
            search_nav: Vec::new(),
            search_cursor: 0,
            prev_search: String::new(),
            selection: None,
            finished_sel: None,
            finished_sel_events: std::collections::HashSet::new(),
            selection_stats: Vec::new(),
            sel_gpu: Vec::new(),
            selection_dirty: false,
            collapsed: BitSet::new(),
            track_scales: Vec::new(),
            even_spacing: false,
            geom: PaneGeom::default(),
            hidden_names: BitSet::new(),
            merge_cache_key: None,
            merged_gpu_groups: Vec::new(),
            exec_context_names: Vec::new(),
            steps: Vec::new(),
            step_cursor: 0,
            gaps: None,
            focused_gap: None,
            gap_filter: GapFilter::default(),
            sel_scopes: Vec::new(),
            pending_tab: Some(BottomTab::Selection),
            bottom_tab: BottomTab::Selection,
            #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
            pending_parse: None,
            #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
            pending_parse_painted: false,
            pending_focus: None,
            sort_col: 2,
            sort_asc: false,
            sort_cache_key: None,
            sort_idx: Vec::new(),
            detail_hist_key: None,
            detail_hist_bins: Vec::new(),
            detail_hist_min: 0.0,
            detail_hist_max: 0.0,
            detail_hist_mean: 0.0,
            detail_hist_median: 0.0,
            sel_aggregate: true,
            label_w: LABEL_W,
            sel_median: 0.0,
            sel_agg_stats: Vec::new(),
            sel_agg_starts: Vec::new(),
            sel_individual: Vec::new(),
            sel_individual_refs: Vec::new(),
            sel_overlap: false,
            sel_overlap_totals: Vec::new(),
            sel_overlap_gen: 0,
            sel_generation: 0,
            track_order: Vec::new(),
            auto_reload: false,
            reload_paths: Vec::new(),
            reload_dir: None,
            cache_dir: None,
            loading_events: Arc::new(AtomicUsize::new(0)),
            #[cfg(not(target_arch = "wasm32"))]
            export_message: None,
            #[cfg(not(target_arch = "wasm32"))]
            share_link_job: None,
            #[cfg(not(target_arch = "wasm32"))]
            share_link_result: None,
            #[cfg(not(target_arch = "wasm32"))]
            share_include_cpu: false,
            sync_message: None,
            color_by_dur: false,
            heat_bounds: None,
        }
    }

    pub fn has_trace(&self) -> bool { self.trace.is_some() }

    /// "File > Export Merged TV Cache": writes this pane's trace to the
    /// on-disk cache that opening a trace used to build automatically (it
    /// no longer does — see `loader::cache_dir_for_folder`). For a
    /// folder-opened pane that's `<folder>.tvcache/_merged.tvcache`; for a
    /// bare single-file open it's a sibling `<file>.tvcache` (same format,
    /// read back by `load_cache`). Already-cached sources (.tvcache family)
    /// have nothing to export.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_tv_cache(&self) -> Result<String, String> {
        let trace = self.trace.as_ref().ok_or_else(|| "no trace loaded".to_string())?;
        if let Some(cd) = &self.cache_dir {
            crate::loader::export_merged_cache(trace, cd)
        } else if self.reload_paths.len() == 1 {
            let path = &self.reload_paths[0].1;
            if path.ends_with(".tvcache") || path.ends_with(".tvcache.xz") || path.ends_with(".tvcache.gz") {
                Err("this trace was loaded from a cache file - nothing to export".to_string())
            } else {
                crate::loader::export_single_cache(trace, path)
            }
        } else {
            Err("no folder or single source file to export a cache for".to_string())
        }
    }

    /// Sibling-folder naming shared by `export_gpu_only_web`/`export_full_web`:
    /// derived from the source path (or the merge directory's own name for a
    /// multi-rank trace, since `reload_paths` for those points at individual
    /// rank files, not the folder the user actually opened).
    #[cfg(not(target_arch = "wasm32"))]
    fn export_web_paths(&self) -> Result<(std::path::PathBuf, String), String> {
        let sample_path = self.reload_paths.first().map(|(_, p)| p.as_str())
            .ok_or_else(|| "no source file path available for this trace".to_string())?;
        let base = std::path::Path::new(sample_path);
        let parent = base.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf();

        let stem = match &self.reload_dir {
            Some(dir) => std::path::Path::new(dir).file_name()
                .and_then(|s| s.to_str()).unwrap_or("trace").to_string(),
            None => {
                let s = base.file_stem().and_then(|s| s.to_str()).unwrap_or("trace");
                s.strip_suffix(".json").or_else(|| s.strip_suffix(".tar")).unwrap_or(s).to_string()
            }
        };
        Ok((parent, stem))
    }

    /// Exports this pane's GPU tracks (timings only, args stripped) via
    /// `loader::export_gpu_only_web` (plain layout, gzip instead of xz) so
    /// the result can be uploaded somewhere with permissive CORS and opened
    /// through the web build's `?src=<url>` shareable-link loader, which has
    /// no LZMA decoder and can't reverse the xz format's columnar/kernel-
    /// grouped encoding.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_gpu_only_web(&self) -> Result<String, String> {
        let trace = self.trace.as_ref().ok_or_else(|| "no trace loaded".to_string())?;
        let (parent, stem) = self.export_web_paths()?;
        let out_path = parent.join(format!("{stem}-gpu-only")).join("gpu-only-web.tvcache.gz");
        let out_str = out_path.to_str().ok_or_else(|| "non-UTF-8 destination path".to_string())?;
        crate::loader::export_gpu_only_web(trace, out_str)?;
        Ok(out_path.to_string_lossy().into_owned())
    }

    /// Exports the whole trace (CPU and GPU tracks, args left intact — see
    /// `loader::export_full_web`) rather than `export_gpu_only_web`'s
    /// deliberately stripped-down version. Much larger, but complete —
    /// for when you want to share the actual trace, not just GPU timings.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_full_web(&self) -> Result<String, String> {
        let trace = self.trace.as_ref().ok_or_else(|| "no trace loaded".to_string())?;
        let (parent, stem) = self.export_web_paths()?;
        let out_path = parent.join(format!("{stem}-full-export")).join("full-trace.tvcache");
        let out_str = out_path.to_str().ok_or_else(|| "non-UTF-8 destination path".to_string())?;
        crate::loader::export_full_web(trace, out_str)?;
        Ok(out_path.to_string_lossy().into_owned())
    }

    /// Corrects inter-node clock skew across this pane's merged multi-rank
    /// trace — see `loader::sync_multi_rank_clocks` for the full algorithm
    /// (a Rust port of `sync_traces.py`, applied in memory instead of
    /// writing shifted files to disk). Only meaningful for an actual
    /// multi-rank merge. Deliberately leaves the view's pan/zoom untouched —
    /// the user is likely already looking at the region they care about
    /// (e.g. the very marker event this aligns on) and a shift on the order
    /// of hundreds of microseconds shouldn't visibly move anything at
    /// whatever zoom level made that visible in the first place.
    ///
    /// Prefers `reload_paths` (the live per-rank source files, when this
    /// pane was opened as a directory/rank list) but falls back to the
    /// merged trace's own `rank_paths` — the same rank/filename pairs,
    /// persisted into the `.tvcache` format at export time — so this still
    /// works after reopening an already-merged cache file directly, when
    /// there's no directory of per-rank JSON to re-derive them from — the
    /// latter is the only path available on wasm (`reload_paths` is always
    /// empty there), but it's enough: any `.tvcache` merged natively and
    /// then opened in the browser carries its `rank_paths` with it.
    pub fn sync_clocks(&mut self) -> Result<String, String> {
        self.sync_clocks_on(crate::loader::DEFAULT_SYNC_MARKER)
    }

    /// Same alignment as `sync_clocks`, but on an arbitrary marker instead of
    /// the hardcoded DeepEP combine kernel — used by the timeline's
    /// right-click "sync ranks to this event" context menu, where `marker`
    /// is the exact name of whichever kernel instance the user clicked.
    pub fn sync_clocks_on(&mut self, marker: &str) -> Result<String, String> {
        let rank_paths: Vec<(usize, String)> = if self.reload_paths.len() >= 2 {
            self.reload_paths.clone()
        } else {
            let trace = self.trace.as_ref().ok_or_else(|| "no trace loaded".to_string())?;
            if trace.rank_paths.len() < 2 {
                return Err("not a multi-rank trace".to_string());
            }
            trace.rank_paths.clone()
        };
        let trace = self.trace.as_mut().ok_or_else(|| "no trace loaded".to_string())?;
        let result = crate::loader::sync_multi_rank_clocks(trace, &rank_paths, marker);
        // Shifts event timestamps in place (same events/indices, so this
        // isn't the out-of-bounds risk a reload is) but the merged view's
        // cached Tetris packing is order-dependent on those timestamps, and
        // the cache key doesn't otherwise change just because clocks synced.
        // Step boundaries are derived from the same timestamps.
        if result.is_ok() {
            self.merge_cache_key = None;
            self.recompute_steps();
            // Shifted timestamps move every gap boundary.
            self.gaps = None;
            self.focused_gap = None;
            // The filter window was picked in pre-shift timestamps.
            self.gap_filter.window = None;
        }
        result
    }

    /// Exports (via `export_gpu_only_web`, or `export_full_web` when
    /// `include_cpu` is set) and, on a background thread, uploads the
    /// result to a new secret GitHub gist through the locally authenticated
    /// `gh` CLI — no browser OAuth flow or pasted token involved, since
    /// this only runs on native where `gh`'s own stored credentials are
    /// already available. The export itself is fast (local disk only) and
    /// stays synchronous; only the network upload (`loader::upload_gist`)
    /// is backgrounded, since that can take a few seconds and shouldn't
    /// freeze the UI.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_share_upload(&mut self, include_cpu: bool) {
        if self.share_link_job.is_some() { return; }
        let path = match if include_cpu { self.export_full_web() } else { self.export_gpu_only_web() } {
            Ok(p) => p,
            Err(e) => { self.share_link_result = Some((false, e)); return; }
        };
        let (tx, rx) = mpsc::channel();
        self.share_link_job = Some(rx);
        self.share_link_result = Some((true, "Uploading to gist...".to_string()));
        rayon::spawn(move || {
            let _ = tx.send(crate::loader::upload_gist(&path));
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn poll_share_upload(&mut self) {
        let rx = match &self.share_link_job {
            Some(rx) => rx,
            None => return,
        };
        match rx.try_recv() {
            Ok(Ok(link)) => {
                self.share_link_result = Some((true, link));
                self.share_link_job = None;
            }
            Ok(Err(e)) => {
                self.share_link_result = Some((false, e));
                self.share_link_job = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => { self.share_link_job = None; }
        }
    }

    pub fn loading_progress_text(&self) -> String {
        if let Some(text) = self.stepped_status() {
            return text;
        }
        let n = self.loading_events.load(Ordering::Relaxed);
        if n == 0 {
            "Loading: reading file...".to_string()
        } else if n < 1_000 {
            format!("Loading: {} events...", n)
        } else if n < 1_000_000 {
            format!("Loading: {:.1}K events...", n as f64 / 1_000.0)
        } else {
            format!("Loading: {:.2}M events...", n as f64 / 1_000_000.0)
        }
    }

    /// The stepped wasm load's phase text (decompress/merge/assemble/
    /// compact), which the event counter can't describe. None during the
    /// parse phase — the counter animates that one.
    #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
    fn stepped_status(&self) -> Option<String> {
        match &self.pending_parse {
            Some(PendingParse::Stepping { plan, .. }) => plan.status(),
            _ => None,
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "mt"))]
    fn stepped_status(&self) -> Option<String> {
        None
    }

    pub fn clear_selection(&mut self) {
        self.selection = None;
        self.finished_sel = None;
        self.finished_sel_events.clear();
        self.selection_stats.clear();
        self.sel_gpu.clear();
        self.sel_scopes.clear();
        // The Gaps tab was scoped to the selection; without one it falls
        // back to the manual filter, so the cached list is stale.
        self.gaps = None;
        self.focused_gap = None;
        self.sel_median = 0.0;
        self.sel_agg_stats.clear();
        self.sel_agg_starts.clear();
        self.sel_individual.clear();
        self.sel_individual_refs.clear();
    }

    pub fn clear_search(&mut self) {
        self.search.clear();
        self.prev_search.clear();
        self.search_mask.clear();
        self.search_nav.clear();
        self.search_cursor = 0;
    }

    pub fn finish_selection(&mut self, buf: &mut DrawBuf) {
        self.finished_sel = self.selection;
        self.finished_sel_events = self.selection.map(|s| self.capture_sel_events(s).into_iter().collect()).unwrap_or_default();
        // Freeze the rank scopes the rect crossed too (same layout-freezing
        // rationale), and invalidate the cached gaps: they're scoped to the
        // selection, so a new selection means a new list.
        self.sel_scopes = self.selection.map(|s| self.capture_sel_scopes(s)).unwrap_or_default();
        self.gaps = None;
        self.focused_gap = None;
        self.selection = None;
        self.rebuild_selection_stats(buf);
    }

    /// Resolve a raw pixel-space selection rectangle into the concrete set of
    /// `(track_idx, event_idx)` pairs it covers, using `self.geom`/
    /// `hidden_names`/`show_cpu` exactly as they are right now (drag-finish
    /// time) — full precision, including which merged-row depth lane the drag
    /// covered for events straddling its time-span edges (events fully inside
    /// the span are captured lane-agnostically; see the merged branch below).
    /// The result is frozen from this point on: nothing that happens
    /// afterward (Show CPU toggle, track reorder/resize, hiding a name) can
    /// retroactively add, remove, or reassign what this selection refers to.
    /// That freezing is the fix — the old code re-derived "what's selected"
    /// from the raw Y range against whatever the CURRENT layout was on every
    /// read, which silently changed the answer as the layout changed.
    pub(crate) fn capture_sel_events(&self, sel: [f64; 4]) -> Vec<(u32, u32)> {
        let trace = match &self.trace { Some(t) => t, None => return Vec::new() };
        let (s0, s1) = if sel[0] <= sel[1] { (sel[0], sel[1]) } else { (sel[1], sel[0]) };
        let (y0, y1) = if sel[2] <= sel[3] { (sel[2] as f32, sel[3] as f32) } else { (sel[3] as f32, sel[2] as f32) };
        let mut out = Vec::new();
        for vi in 0..self.geom.visible.len() {
            let track_top = self.geom.y_offsets[vi];
            let track_h = self.geom.heights[vi];
            let track_bot = track_top + track_h;
            if track_bot < y0 || track_top > y1 { continue; }

            if let Some(group) = self.geom.merged.iter().find(|g| g.vi == vi) {
                // Match the rendered merged row: iterate the packed events
                // (wrappers stripped) and apply the renderer's depth/y test.
                // Same windowing the render loop uses (pack is whole-span,
                // view-independent — see MergedPack).
                let p = &group.pack;
                let max_depth = p.max_depth.max(1);
                let sub_h = track_h / max_depth as f32;
                let start = p.overlap_start(s0);
                let end = p.ts_end(s1);
                for gi in start..end {
                    let (ti32, ei32) = p.refs[gi];
                    let depth = p.depth[gi];
                    let ts = p.ts[gi];
                    let end_t = p.ends[gi];
                    if !(end_t >= s0 && ts <= s1) { continue; }
                    // A merged row's depth lanes are a rendering artifact
                    // (Tetris packing against overdraw), not a semantic
                    // dimension: events sharing a lane are disjoint BY
                    // CONSTRUCTION, so a thin horizontal band crossing one
                    // or two lanes captures an arbitrary ~1/max_depth slice
                    // of the rank's kernels — which then cannot overlap,
                    // silently deflating the Selection tab's counts/sums
                    // and making the overlap-aware totals equal the plain
                    // sums (the toggle "did nothing"). An event FULLY
                    // INSIDE the dragged time span is therefore captured
                    // wherever its lane sits ("this time range on this
                    // rank"). Only events straddling the span's edges keep
                    // lane precision below — which is also what keeps a
                    // shift-click (a zero-width rect that can't contain
                    // anything) selecting exactly the one bar under the
                    // cursor.
                    if !(ts >= s0 && end_t <= s1) {
                        // Same stretch bounds the renderer draws (and the
                        // live drag tests): a kernel stretched down into
                        // empty depth lanes must be captured when the drag
                        // covers the lower (stretched) part of its bar, not
                        // just its own lane. Same stale-cache guard the
                        // render loop applies.
                        let (lo, hi) = p.stretch.get(gi).copied().unwrap_or((depth, depth));
                        let ev_top = track_top + lo as f32 * sub_h;
                        let ev_bot = ev_top + (hi - lo + 1) as f32 * sub_h;
                        if ev_bot < y0 || ev_top > y1 { continue; }
                    }
                    out.push((ti32, ei32));
                }
            } else {
                let ti = self.geom.visible[vi];
                let track = &trace.tracks[ti];
                if !self.show_cpu && !track.gpu { continue; }
                // Mirror the renderer exactly: a collapsed row is ONE lane
                // showing only depth-0 events across its full height, so
                // deeper events are invisible and must not be captured.
                let is_collapsed = self.collapsed.get(ti).unwrap_or(false);
                let sub_h = track_h / if is_collapsed { 1 } else { track.max_depth.max(1) } as f32;
                let start = bisect_overlap(&track.events, &track.prefix_max_dur, s0);
                let end = track.events.partition_point(|e| e.ts <= s1).max(start);
                for (local_i, ev) in track.events[start..end].iter().enumerate() {
                    if self.hidden_names.get(ev.name as usize).unwrap_or(false) { continue; }
                    if is_collapsed && ev.depth > 0 { continue; }
                    let ev_top = track_top + ev.depth as f32 * sub_h;
                    let ev_bot = ev_top + sub_h;
                    if ev_bot < y0 || ev_top > y1 { continue; }
                    if ev.ts + ev.dur >= s0 && ev.ts <= s1 {
                        out.push((ti as u32, (start + local_i) as u32));
                    }
                }
            }
        }
        out
    }

    /// Rank scopes of every visible row the selection rect's y-range
    /// crosses, resolved against the layout at drag-finish time — same
    /// freezing rationale as `capture_sel_events` (re-deriving from the raw
    /// y range against a later layout would silently change the answer as
    /// rows are reordered/resized/toggled). Merged rows contribute every
    /// packed track: depth lanes are a rendering artifact, not a semantic
    /// dimension (see `capture_sel_events`). Keyed the way
    /// `compute_gaps_in` scopes gaps: `parse_rank(...).unwrap_or(MAX)`.
    pub(crate) fn capture_sel_scopes(&self, sel: [f64; 4]) -> Vec<usize> {
        let trace = match &self.trace { Some(t) => t, None => return Vec::new() };
        let (y0, y1) = if sel[2] <= sel[3] { (sel[2] as f32, sel[3] as f32) } else { (sel[3] as f32, sel[2] as f32) };
        let mut out: std::collections::HashSet<usize> = Default::default();
        for vi in 0..self.geom.visible.len() {
            let track_top = self.geom.y_offsets[vi];
            let track_bot = track_top + self.geom.heights[vi];
            if track_bot < y0 || track_top > y1 { continue; }
            if let Some(group) = self.geom.merged.iter().find(|g| g.vi == vi) {
                for &(ti, _) in &group.pack.refs {
                    out.insert(parse_rank(&trace.tracks[ti as usize].label).unwrap_or(usize::MAX));
                }
            } else {
                let ti = self.geom.visible[vi];
                let track = &trace.tracks[ti];
                if !self.show_cpu && !track.gpu { continue; }
                out.insert(parse_rank(&track.label).unwrap_or(usize::MAX));
            }
        }
        let mut v: Vec<usize> = out.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// The finished selection's time range when it should take over gap
    /// detection: non-zero-width drag selections only (a shift-click's
    /// zero-width rect picks one event and would clip every gap away).
    /// None = no selection scoping, use the manual `gap_filter` window.
    pub fn gap_sel_window(&self) -> Option<(f64, f64)> {
        let sel = self.finished_sel?;
        let (s0, s1) = if sel[0] <= sel[1] { (sel[0], sel[1]) } else { (sel[1], sel[0]) };
        (s1 > s0).then_some((s0, s1))
    }

    pub fn open(&mut self, path: String) {
        let (tx, rx) = mpsc::channel();
        self.loading = Some(rx);
        self.error = None;
        self.trace_path = path.clone();
        self.reload_paths = vec![(0, path.clone())];
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        let cd = self.cache_dir.clone();
        spawn_load_job(move || {
            load_trace_progressive(&path, &counter, 0, &tx, cd.as_deref());
        });
    }

    /// wasm equivalent of `open`: takes bytes already read into memory (e.g.
    /// via the browser's drag-and-drop/file-picker File API) instead of a
    /// filesystem path — there's no mmap/filesystem on wasm32 to open a path
    /// against. `name` is the dropped/picked file's name (e.g.
    /// `File::name()`), used only to sniff the format (.tvcache/.tar.gz/.gz
    /// vs plain JSON), same role a real path's suffix plays natively.
    #[cfg(target_arch = "wasm32")]
    pub fn open_from_bytes(&mut self, name: String, bytes: Vec<u8>) {
        let (tx, rx) = mpsc::channel();
        self.loading = Some(rx);
        self.error = None;
        self.trace_path = name.clone();
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        // mt: spawn_load_job puts the parse on the rayon Worker pool, so it
        // can start immediately. Single-threaded: it would run inline right
        // here, before any frame could paint a "Parsing trace..." splash —
        // stage it for `run_pending_parse` instead (one frame later).
        #[cfg(feature = "mt")]
        spawn_load_job(move || {
            crate::loader::load_trace_from_bytes_progressive(&name, bytes, &counter, &tx);
        });
        #[cfg(not(feature = "mt"))]
        {
            crate::tlog!("staged: {name} ({} bytes)", bytes.len());
            self.pending_parse = Some(PendingParse::Single { name, bytes, counter, tx });
            self.pending_parse_painted = false;
        }
    }

    /// wasm equivalent of `open_multi`: each rank's bytes are already in
    /// memory (e.g. from walking a dropped folder via the browser's File and
    /// Directory Entries API — see main.rs), so there's no path to re-read
    /// from, and no `reload_dir`/`reload_paths` equivalent since a browser
    /// drop can't be "re-opened" the way a filesystem directory can.
    #[cfg(target_arch = "wasm32")]
    pub fn open_multi_from_bytes(&mut self, rank_named_bytes: Vec<(usize, String, Vec<u8>)>) {
        let (tx, rx) = mpsc::channel();
        let n = rank_named_bytes.len();
        let fname = rank_named_bytes[0].1.rsplit('/').next().unwrap_or(&rank_named_bytes[0].1);
        let prefix = fname.find("-rank-").map(|p| fname[..p].to_string())
            .unwrap_or_else(|| "multi-rank".to_string());
        self.trace_path = format!("{} ranks: {}", n, prefix);
        self.loading = Some(rx);
        self.error = None;
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        // Same mt-vs-single-threaded split as `open_from_bytes`.
        #[cfg(feature = "mt")]
        spawn_load_job(move || {
            crate::loader::load_multi_from_bytes_progressive(rank_named_bytes, &counter, &tx);
        });
        #[cfg(not(feature = "mt"))]
        {
            self.pending_parse = Some(PendingParse::Multi { rank_named_bytes, counter, tx });
            self.pending_parse_painted = false;
        }
    }

    /// Single-threaded wasm build: advance a staged load by one frame's
    /// worth of work (see `PendingParse`). The first frame after staging
    /// only sets `pending_parse_painted`, so its splash ("Parsing
    /// trace...") paints before the main thread does any blocking work.
    /// After that: prepare the plan, then step a time-budgeted slice of it
    /// per frame (the event counter animates during the parse phase), then
    /// finish — which sends through the channel, so `poll_loading` (called
    /// right after this in render_frame) picks the result up in the same
    /// frame.
    #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
    pub(crate) fn run_pending_parse(&mut self) {
        if self.pending_parse.is_none() {
            return;
        }
        if !self.pending_parse_painted {
            self.pending_parse_painted = true;
            return;
        }
        let job = match self.pending_parse.take() {
            Some(j) => j,
            None => return,
        };
        self.pending_parse = match job {
            PendingParse::Single { name, bytes, counter, tx } => {
                match crate::loader::WasmParsePlan::prepare(&name, bytes, &counter, true) {
                    crate::loader::Prepared::Done(result) => {
                        let _ = tx.send(result);
                        None
                    }
                    crate::loader::Prepared::Plan(plan) => {
                        Some(PendingParse::Stepping { plan, tx })
                    }
                }
            }
            PendingParse::Multi { rank_named_bytes, counter, tx } => {
                match crate::loader::WasmMultiParsePlan::prepare(rank_named_bytes, &counter) {
                    crate::loader::Prepared::Done(result) => {
                        let _ = tx.send(result);
                        None
                    }
                    crate::loader::Prepared::Plan(plan) => {
                        Some(PendingParse::Stepping { plan, tx })
                    }
                }
            }
            PendingParse::Stepping { mut plan, tx } => {
                if plan.step() {
                    plan.finish(&tx);
                    None
                } else {
                    Some(PendingParse::Stepping { plan, tx })
                }
            }
        };
    }

    /// Whether a staged load is still waiting for its first frame (not yet
    /// prepared or stepping) — the splash shows "Parsing trace..." for
    /// this case and the animated event count once stepping begins.
    #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
    pub(crate) fn parse_announced(&self) -> bool {
        matches!(
            self.pending_parse,
            Some(PendingParse::Single { .. }) | Some(PendingParse::Multi { .. })
        )
    }

    /// Load-in-flight status for the web boot overlay (update_boot_overlay
    /// in main.rs): `None` when this pane isn't loading, `Some(None)` when
    /// it is but no fraction is known (staged-not-yet-stepping, or the mt
    /// build's pool-side parse), `Some(Some(f))` while a stepped parse can
    /// report its phase progress.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn boot_parse_progress(&self) -> Option<Option<f32>> {
        #[cfg(not(feature = "mt"))]
        if let Some(PendingParse::Stepping { plan, .. }) = &self.pending_parse {
            return Some(Some(plan.progress()));
        }
        if self.loading.is_some() { Some(None) } else { None }
    }

    pub fn open_multi(&mut self, rank_paths: Vec<(usize, String)>) {
        let (tx, rx) = mpsc::channel();
        let n = rank_paths.len();
        let prefix = std::path::Path::new(&rank_paths[0].1)
            .file_name()
            .and_then(|f| f.to_str())
            .and_then(|f| f.find("-rank-").map(|p| &f[..p]))
            .unwrap_or("multi-rank")
            .to_string();
        self.trace_path = format!("{} ranks: {}", n, prefix);
        self.reload_paths = rank_paths.clone();
        self.loading = Some(rx);
        self.error = None;
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        let tpf = (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / n).max(2);
        let cd = self.cache_dir.clone();
        spawn_load_job(move || {
            load_multi_progressive(rank_paths, &counter, tpf, &tx, cd.as_deref(), false);
        });
    }

    pub fn reload(&mut self) {
        if self.loading.is_some() { return; }
        if let Some(dir) = &self.reload_dir {
            let (groups, standalone) = crate::loader::detect_rank_groups(&[dir.clone()]);
            let mut all_paths: Vec<(usize, String)> = Vec::new();
            for (group, _dir) in groups {
                all_paths.extend(group);
            }
            for (i, (path, _dir)) in standalone.into_iter().enumerate() {
                let rank = all_paths.len() + i;
                all_paths.push((rank, path));
            }
            if all_paths.is_empty() { return; }
            self.reload_paths = all_paths;
        }
        if self.reload_paths.is_empty() { return; }
        let (tx, rx) = mpsc::channel();
        self.loading = Some(rx);
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        let paths = self.reload_paths.clone();
        let cd = self.cache_dir.clone();
        if paths.len() == 1 {
            let path = paths[0].1.clone();
            spawn_load_job(move || {
                load_trace_progressive(&path, &counter, 0, &tx, cd.as_deref());
            });
        } else {
            let tpf = (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / paths.len()).max(1);
            spawn_load_job(move || {
                load_multi_progressive(paths, &counter, tpf, &tx, cd.as_deref(), true);
            });
        }
    }

    pub fn poll_loading(&mut self) {
        let rx = match &self.loading {
            Some(rx) => rx,
            None => return,
        };
        match rx.try_recv() {
            Ok(Ok(trace)) => {
                crate::tlog!("arrived: {} tracks, {} events", trace.tracks.len(), trace.total_events);
                let is_reload = self.trace.is_some();
                // Heat-map bounds are a function of the trace's durations;
                // a new (or reloaded, possibly changed) trace invalidates
                // them, and gap timestamps don't survive the swap either.
                self.heat_bounds = None;
                self.gaps = None;
                self.focused_gap = None;
                // New trace, new timestamps: the filter window is stale.
                self.gap_filter.window = None;
                if is_reload {
                    let n_tracks = trace.tracks.len();
                    let n_names = trace.names.len();
                    self.collapsed.resize(n_tracks);
                    self.track_scales.resize(n_tracks, 1.0);
                    if self.track_order.len() != n_tracks {
                        self.track_order = default_track_order(&trace.tracks);
                    }
                    self.hidden_names.resize(n_names);
                    self.search_mask.clear();
                    self.search_nav.clear();
                    self.search_cursor = 0;
                    // The reloaded trace may have fewer events; a retained
                    // EventRef would index out of bounds in the Detail panel,
                    // and the finished selection's frozen event set has the
                    // same hazard in every Selection-tab consumer
                    // (rebuild_multi_stats, sel_events_list,
                    // copy_selection_text, selection_gpu_summary) — they all
                    // index tracks/events directly. clear_selection() drops
                    // the whole selection-derived family at once.
                    self.selected = None;
                    self.clear_selection();
                    // Same reasoning: the merged-view Tetris-packing cache
                    // stores raw (track_idx, event_idx) pairs keyed on
                    // (view range, hidden names, track order) — none of
                    // which necessarily change on a reload, so a stale cache
                    // hit would keep pointing at indices that no longer
                    // exist in the new trace's (possibly smaller) tracks.
                    self.merge_cache_key = None;
                    let old_stats = std::mem::take(&mut self.trace.as_mut().unwrap().stats);
                    self.trace = Some(trace);
                    self.trace.as_mut().unwrap().stats = old_stats;
                    if !self.search.is_empty() { self.rebuild_search(); }
                } else {
                    let pad = trace.max_ts * 0.02;
                    self.view.t0 = -pad;
                    self.view.t1 = trace.max_ts + pad;
                    self.view.scroll_y = 0.0;
                    self.collapsed = BitSet::with_len(trace.tracks.len());
                    self.track_scales = vec![1.0; trace.tracks.len()];
                    self.track_order = default_track_order(&trace.tracks);
                    self.hidden_names = BitSet::with_len(trace.names.len());
                    self.trace = Some(trace);
                }
                // Computed once per load instead of every toolbar frame — the
                // "Hide Execute Context" button's name list only ever needs
                // this trace's `names`, which don't change until the next load.
                self.exec_context_names = self.trace.as_ref()
                    .map(|t| find_exec_context_names(&t.names))
                    .unwrap_or_default();
                self.recompute_steps();
            }
            Ok(Err(e)) => {
                self.error = Some(e);
                self.loading = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.loading = None;
            }
        }
    }

    pub fn rebuild_search(&mut self) {
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let q = self.search.trim().to_ascii_lowercase();
        self.search_mask.clear();
        self.search_mask.resize(trace.names.len());
        self.search_nav.clear();
        self.search_cursor = 0;

        if q.is_empty() { return; }

        for (i, name) in trace.names.iter().enumerate() {
            if name.to_ascii_lowercase().contains(&q) {
                self.search_mask.set(i, true);
            }
        }

        for (ti, track) in trace.tracks.iter().enumerate() {
            for (ei, ev) in track.events.iter().enumerate() {
                if self.search_mask[ev.name as usize] {
                    self.search_nav.push((ev.ts, ti as u32, ei as u32));
                }
            }
        }
        self.search_nav.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    pub fn rebuild_selection_stats(&mut self, buf: &mut DrawBuf) {
        let t_start = crate::time::Instant::now();
        if self.trace.is_none() { return; }
        if self.finished_sel.is_none() {
            self.selection_stats.clear(); self.sel_gpu.clear(); self.sel_agg_stats.clear(); self.sel_agg_starts.clear(); self.sel_individual.clear(); self.sel_individual_refs.clear(); self.sel_median = 0.0;
            return;
        }
        // finished_sel_events is the frozen (track_idx, event_idx) set
        // resolved when the drag finished (see capture_sel_events) — reading
        // it directly instead of re-deriving track membership from the raw
        // selection rectangle keeps this stable across later layout changes
        // (Show CPU toggle, track reorder/resize).
        let trace = self.trace.as_ref().unwrap();

        let map = &mut buf.sel_map;
        for v in map.values_mut() { v.0 = 0; v.1 = 0.0; v.2.clear(); v.3.clear(); }

        for &(ti, ei) in &self.finished_sel_events {
            let ev = &trace.tracks[ti as usize].events[ei as usize];
            let e = map.entry(ev.name).or_insert((0, 0.0, Vec::new(), Vec::new()));
            e.0 += 1;
            e.1 += ev.dur;
            e.2.push(ev.dur);
            e.3.push((ti, ei));
        }
        self.selection_stats.clear();
        for (&name, (count, total_dur, durations, event_refs)) in map.iter_mut() {
            if *count == 0 { continue; }
            self.selection_stats.push(SelectionEntry {
                name, count: *count, total_dur: *total_dur,
                durations: std::mem::take(durations),
                event_refs: std::mem::take(event_refs),
            });
        }
        self.selection_stats.sort_unstable_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap().then(a.name.cmp(&b.name)));
        // GPU empty-space summary over the same frozen event set. Scoped so
        // the `trace`/`finished_sel_events` borrows end before the assign.
        self.sel_gpu = {
            let trace = self.trace.as_ref().unwrap();
            selection_gpu_summary(trace, &self.finished_sel_events, self.finished_sel.unwrap())
        };
        let ev_count: u32 = self.selection_stats.iter().map(|e| e.count).sum();
        eprintln!("  select: {:.1}ms ({} events, {} names)", t_start.elapsed().as_secs_f64() * 1000.0, ev_count, self.selection_stats.len());

        let t_agg = crate::time::Instant::now();
        self.compute_aggregates();
        eprintln!("  aggregate: {:.1}ms ({} agg, {} individual)", t_agg.elapsed().as_secs_f64() * 1000.0, self.sel_agg_stats.len(), self.sel_individual.len());
    }

    pub fn extract_selection_events(&self) -> Vec<(String, f64)> {
        let trace = match &self.trace {
            Some(t) => t,
            None => return Vec::new(),
        };
        if self.finished_sel.is_none() { return Vec::new(); }

        let mut events: Vec<(f64, String, f64)> = self.finished_sel_events.iter().map(|&(ti, ei)| {
            let ev = &trace.tracks[ti as usize].events[ei as usize];
            (ev.ts, trace.names[ev.name as usize].clone(), ev.dur)
        }).collect();
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        events.into_iter().map(|(_, name, dur)| (name, dur)).collect()
    }

    pub fn copy_selection_text(&self) -> Option<String> {
        let trace = self.trace.as_ref()?;
        let mut events: Vec<(f64, &str, f64)> = Vec::new();

        if self.finished_sel.is_some() {
            for &(ti, ei) in &self.finished_sel_events {
                let ev = &trace.tracks[ti as usize].events[ei as usize];
                events.push((ev.ts, &trace.names[ev.name as usize], ev.dur));
            }
        } else if let Some(name_id) = self.multi_select_name {
            for track in &trace.tracks {
                if !self.show_cpu && !track.gpu { continue; }
                for ev in &track.events {
                    if ev.name == name_id {
                        events.push((ev.ts, &trace.names[ev.name as usize], ev.dur));
                    }
                }
            }
        } else if let Some(sel) = self.selected {
            if let Some(ev) = trace.tracks.get(sel.track_idx as usize)
                .and_then(|t| t.events.get(sel.event_idx as usize))
            {
                events.push((ev.ts, &trace.names[ev.name as usize], ev.dur));
            }
        }

        if events.is_empty() { return None; }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut out = String::new();
        for (_, name, dur) in &events {
            use crate::ui::write_time;
            out.push_str(name);
            out.push_str(" (");
            write_time(&mut out, *dur);
            out.push_str(")\n");
        }
        Some(out)
    }

    pub fn select_from_search(&mut self, buf: &mut DrawBuf) {
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        if !self.search_mask.any() { return; }

        self.selected = None;
        self.multi_select_name = None;
        self.selection = None;
        self.selection_stats.clear();

        let map = &mut buf.sel_map;
        for v in map.values_mut() { v.0 = 0; v.1 = 0.0; v.2.clear(); v.3.clear(); }

        for (ti, track) in trace.tracks.iter().enumerate() {
            if !self.show_cpu && !track.gpu { continue; }
            for (ei, ev) in track.events.iter().enumerate() {
                if self.hidden_names.get(ev.name as usize).unwrap_or(false) { continue; }
                if self.search_mask[ev.name as usize] {
                    let e = map.entry(ev.name).or_insert((0, 0.0, Vec::new(), Vec::new()));
                    e.0 += 1;
                    e.1 += ev.dur;
                    e.2.push(ev.dur);
                    e.3.push((ti as u32, ei as u32));
                }
            }
        }
        self.selection_stats.clear();
        for (&name, (count, total_dur, durations, event_refs)) in map.iter_mut() {
            if *count == 0 { continue; }
            self.selection_stats.push(SelectionEntry {
                name, count: *count, total_dur: *total_dur,
                durations: std::mem::take(durations),
                event_refs: std::mem::take(event_refs),
            });
        }
        self.selection_stats.sort_unstable_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap().then(a.name.cmp(&b.name)));
        self.compute_aggregates();
    }

    /// Build the Selection-tab stats for a double-click multi-select: every
    /// event whose name matches `multi_select_name`, across the visible tracks
    /// (respecting the CPU/GPU filter), so the tab shows that kernel's count,
    /// totals and duration distribution — matching the timeline highlight.
    pub fn rebuild_multi_select_stats(&mut self) {
        let name_id = match self.multi_select_name {
            Some(n) => n,
            None => return,
        };
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let mut count = 0u32;
        let mut total_dur = 0.0f64;
        let mut durations = Vec::new();
        let mut event_refs = Vec::new();
        for (ti, track) in trace.tracks.iter().enumerate() {
            if !self.show_cpu && !track.gpu { continue; }
            for (ei, ev) in track.events.iter().enumerate() {
                if ev.name != name_id { continue; }
                count += 1;
                total_dur += ev.dur;
                durations.push(ev.dur);
                event_refs.push((ti as u32, ei as u32));
            }
        }
        self.selection_stats.clear();
        if count > 0 {
            self.selection_stats.push(SelectionEntry { name: name_id, count, total_dur, durations, event_refs });
        }
        self.compute_aggregates();
    }

    /// Start a smooth zoom to the FIRST (earliest) search match, sized so that
    /// event fills `SEARCH_ZOOM_FILL` (80%) of the timeline width, and record
    /// its track as the pending vertical focus so draw_timeline can scroll it
    /// into view. Uses the same visibility filters as `select_from_search`.
    /// Whether the match at `(track_idx, event_idx)` is currently on screen —
    /// i.e. its track isn't CPU-hidden and its name isn't hidden. Search
    /// navigation skips matches that fail this, so it never frames an event on a
    /// track the user can't see.
    fn nav_match_visible(&self, ti: u32, ei: u32) -> bool {
        let trace = match &self.trace { Some(t) => t, None => return false };
        let track = match trace.tracks.get(ti as usize) { Some(t) => t, None => return false };
        if !self.show_cpu && !track.gpu { return false; }
        match track.events.get(ei as usize) {
            Some(ev) => !self.hidden_names.get(ev.name as usize).unwrap_or(false),
            None => false,
        }
    }

    /// Frame the first visible match and sync the nav cursor to it, so a
    /// subsequent prev/next continues from the framed event rather than from a
    /// stale cursor (which could step into a hidden CPU match and appear to do
    /// nothing).
    pub fn zoom_to_search(&mut self) {
        let idx = self.search_nav.iter()
            .position(|&(_, ti, ei)| self.nav_match_visible(ti, ei));
        if let Some(idx) = idx {
            self.search_cursor = idx;
            self.zoom_to_nav_cursor();
        }
    }

    fn zoom_to_nav_cursor(&mut self) {
        let (ts, ti, ei) = self.search_nav[self.search_cursor];
        self.selected = Some(EventRef { track_idx: ti, event_idx: ei });
        let dur = self.trace.as_ref().unwrap().tracks[ti as usize].events[ei as usize].dur;
        self.start_zoom_to(ts, dur, ti as usize);
    }

    /// Rebuild `steps` from the current trace + exec-context name list.
    /// Called after load, clock sync, and auto-reload — the only events that
    /// can shift the timestamps or name set the steps are derived from.
    pub fn recompute_steps(&mut self) {
        self.steps = match &self.trace {
            Some(t) => compute_steps(t, &self.exec_context_names),
            None => Vec::new(),
        };
        if self.step_cursor >= self.steps.len() { self.step_cursor = 0; }
    }

    /// `[`/`]`: step the cursor and smooth-zoom to that step, framed with a
    /// little padding either side so the inter-step gaps stay visible.
    /// Wraps around like search nav does.
    pub fn nav_step(&mut self, forward: bool) {
        let n = self.steps.len();
        if n == 0 { return; }
        // Snap the cursor to the step the view center is in before stepping,
        // so `[` from an arbitrary zoom level goes to the *next* step from
        // where you're looking, not from wherever the cursor last was.
        let center = (self.view.t0 + self.view.t1) / 2.0;
        if let Some(i) = self.steps.iter().position(|s| center < s.t1) {
            self.step_cursor = i;
        }
        self.step_cursor = if forward {
            (self.step_cursor + 1) % n
        } else {
            (self.step_cursor + n - 1) % n
        };
        self.frame_step(self.step_cursor);
    }

    /// Smooth-zoom so step `i` fills the view with `STEP_PAD` slack on each
    /// side. Keeps the current vertical scroll — steps span all tracks, so
    /// there's nothing specific to scroll to.
    pub fn frame_step(&mut self, i: usize) {
        let Some(s) = self.steps.get(i) else { return; };
        let range = ((s.t1 - s.t0) / (1.0 - 2.0 * crate::types::STEP_PAD))
            .max(crate::types::MIN_TIME_RANGE);
        let center = (s.t0 + s.t1) / 2.0;
        self.view.anim = Some(crate::types::ViewAnim {
            from_t0: self.view.t0,
            from_t1: self.view.t1,
            to_t0: center - range / 2.0,
            to_t1: center + range / 2.0,
            from_scroll: self.view.scroll_y,
            to_scroll: self.view.scroll_y,
            elapsed: 0.0,
            dur: crate::types::ZOOM_ANIM_DUR,
        });
    }

    /// Smooth-zoom to [t0, t1] with the same `STEP_PAD` framing as
    /// `frame_step`, and scroll `focus_track`'s row into view — the Gaps
    /// tab's row-click navigation (the gap belongs to that rank, so its row
    /// is where the eye should land).
    pub fn frame_gap(&mut self, t0: f64, t1: f64, focus_track: u32) {
        let range = ((t1 - t0) / (1.0 - 2.0 * crate::types::STEP_PAD))
            .max(crate::types::MIN_TIME_RANGE);
        let center = (t0 + t1) / 2.0;
        self.view.anim = Some(crate::types::ViewAnim {
            from_t0: self.view.t0,
            from_t1: self.view.t1,
            to_t0: center - range / 2.0,
            to_t1: center + range / 2.0,
            from_scroll: self.view.scroll_y,
            to_scroll: self.view.scroll_y, // resolved in draw_timeline from layout
            elapsed: 0.0,
            dur: crate::types::ZOOM_ANIM_DUR,
        });
        self.pending_focus = Some(focus_track);
    }

    /// Compute `gaps` if dirty (see the field doc). Called from the Gaps
    /// tab body, so the sort cost lands on the first frame the tab is
    /// actually open, never on the load path.
    pub fn ensure_gaps(&mut self) {
        if self.gaps.is_some() { return; }
        // A finished selection takes over detection: gaps are clipped to its
        // time range and scoped to the ranks its rect crossed. Without one
        // (or for a zero-width pick, see gap_sel_window) the manual filter
        // applies.
        let mut f = self.gap_filter.clone();
        if let Some((s0, s1)) = self.gap_sel_window() {
            f.window = Some((s0, s1));
            f.scopes = Some(self.sel_scopes.clone());
        }
        self.gaps = Some(match &self.trace {
            Some(t) => compute_gaps_in(t, f),
            None => Vec::new(),
        });
    }

    /// Start a smooth zoom that frames a single event (`ts`, `dur`) at
    /// `SEARCH_ZOOM_FILL` of the width and requests a vertical scroll to its
    /// track. The vertical target is resolved in `draw_timeline` from layout.
    pub fn start_zoom_to(&mut self, ts: f64, dur: f64, track_idx: usize) {
        let range = (dur / crate::types::SEARCH_ZOOM_FILL).max(crate::types::MIN_TIME_RANGE);
        let center = ts + dur / 2.0;
        let to_t0 = center - range / 2.0;
        let to_t1 = center + range / 2.0;
        self.view.anim = Some(crate::types::ViewAnim {
            from_t0: self.view.t0,
            from_t1: self.view.t1,
            to_t0,
            to_t1,
            from_scroll: self.view.scroll_y,
            to_scroll: self.view.scroll_y, // resolved in draw_timeline from layout
            elapsed: 0.0,
            dur: crate::types::ZOOM_ANIM_DUR,
        });
        self.pending_focus = Some(track_idx as u32);
    }

    /// Step the search-match cursor by one and smooth-zoom to that match.
    /// `forward` advances, `!forward` goes back; both wrap around. Selects the
    /// event and switches the bottom panel to the Detail tab.
    pub fn nav_search(&mut self, forward: bool) {
        let n = self.search_nav.len();
        if n == 0 { return; }
        // Step to the next/previous *visible* match, skipping hidden CPU-track
        // and hidden-name matches, wrapping around. If none are visible, do
        // nothing rather than jump to an off-screen event.
        let mut i = self.search_cursor;
        for _ in 0..n {
            i = if forward { (i + 1) % n } else { (i + n - 1) % n };
            let (_, ti, ei) = self.search_nav[i];
            if self.nav_match_visible(ti, ei) {
                self.search_cursor = i;
                self.pending_tab = Some(BottomTab::Detail);
                self.zoom_to_nav_cursor();
                return;
            }
        }
    }

    /// Compute `sel_overlap_totals` for the current selection if stale: one
    /// row per `selection_stats` entry (same order `compute_aggregates`
    /// builds `sel_agg_stats` in), each the *shared* time of that kernel —
    /// every instant of wall time split equally among the selected kernels
    /// running it, summed over rank scopes. Timestamps come from the trace
    /// via `event_refs` — the `durations` list is duration-only, and
    /// overlap is a timestamp property.
    ///
    /// Even sharing is the discount the toggle exists for: a kernel that
    /// always runs alongside one other is worth half its summed duration
    /// ("less of a problem"), a solo kernel keeps everything, and the
    /// totals sum to the selection's busy wall time — concurrency is never
    /// double-counted. Exclusive time (credit only solo instants) was
    /// tried first and zeroes the critical path in densely-overlapped
    /// windows: everything co-runs by construction, so the only nonzero
    /// rows are straggler tails and the % column is decided by whichever
    /// tiny kernel happened to run alone.
    ///
    /// Scopes: events are grouped by `parse_rank` of their track's label —
    /// the same grouping merged rows use — and each scope swept
    /// independently, so overlap only discounts a kernel against kernels
    /// on its own rank (a rank-synchronous trace would otherwise discount
    /// everything against the other ranks). Tracks without a rank label
    /// share one scope, which is also correct for a single-rank trace.
    pub fn ensure_sel_overlap_totals(&mut self) {
        if self.sel_overlap_gen == self.sel_generation
            && self.sel_overlap_totals.len() == self.selection_stats.len() {
            return;
        }
        let trace = match &self.trace { Some(t) => t, None => return };
        // Bucket selected events by rank scope. usize::MAX is the shared
        // scope for tracks without a rank label.
        let mut buckets: std::collections::HashMap<usize, Vec<(f64, f64, u32)>> =
            Default::default();
        for se in &self.selection_stats {
            for &(ti, ei) in &se.event_refs {
                let Some(ev) = trace.tracks.get(ti as usize)
                    .and_then(|t| t.events.get(ei as usize))
                else { continue };
                if ev.dur <= 0.0 { continue; } // no interval to share
                let scope = parse_rank(&trace.tracks[ti as usize].label)
                    .unwrap_or(usize::MAX);
                buckets.entry(scope).or_default()
                    .push((ev.ts, ev.ts + ev.dur, se.name));
            }
        }
        let mut share: Vec<f64> = vec![0.0; trace.names.len()];
        for (_, events) in buckets {
            // Sweep: points (time, ±1, name), ends sorted before starts at
            // ties so back-to-back events don't count as overlapping.
            // `active` maps name -> live-launch count, so self-overlapping
            // launches keep the name active until its last one ends (a set
            // of names would evict it at the first end); a name's launch
            // count never dilutes its share — the name is credited once
            // per instant, like any other active name.
            let mut points: Vec<(f64, i32, u32)> = Vec::with_capacity(events.len() * 2);
            for &(ts, end, name) in &events {
                points.push((ts, 1, name));
                points.push((end, -1, name));
            }
            points.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut active: std::collections::HashMap<u32, u32> = Default::default();
            let mut prev: Option<f64> = None;
            for &(t, d, name) in &points {
                if let Some(p) = prev {
                    let dt = t - p;
                    if dt > 0.0 && !active.is_empty() {
                        let w = dt / active.len() as f64;
                        for &n in active.keys() {
                            share[n as usize] += w;
                        }
                    }
                }
                if d > 0 {
                    *active.entry(name).or_insert(0) += 1;
                } else if let Some(c) = active.get_mut(&name) {
                    *c -= 1;
                    if *c == 0 { active.remove(&name); }
                }
                prev = Some(t);
            }
        }
        self.sel_overlap_totals = self.selection_stats.iter()
            .map(|se| share[se.name as usize]).collect();
        self.sel_overlap_gen = self.sel_generation;
    }

    /// Compute `heat_bounds` if stale (see the field doc). Sampling keeps
    /// this at ~100k durations regardless of trace size — enough for stable
    /// p5/p95 on a log scale, at a fraction of a full collect+select on the
    /// 1M+ events a large trace carries.
    pub fn ensure_heat_bounds(&mut self) {
        if self.heat_bounds.is_some() { return; }
        let bounds = match &self.trace {
            Some(t) => {
                let n_gpu: usize = t.tracks.iter().filter(|tr| tr.gpu).map(|tr| tr.events.len()).sum();
                if n_gpu == 0 {
                    (0.0, 1.0)
                } else {
                    let stride = (n_gpu / 100_000).max(1);
                    let mut logs: Vec<f64> = Vec::with_capacity(n_gpu / stride + 1);
                    for track in t.tracks.iter().filter(|tr| tr.gpu) {
                        for ev in track.events.iter().step_by(stride) {
                            if ev.dur > 0.0 { logs.push(ev.dur.log10()); }
                        }
                    }
                    if logs.is_empty() {
                        (0.0, 1.0)
                    } else {
                        // Two in-place selections: after the p5 select, the
                        // lower half is all <= logs[i5], so the p95 select can
                        // run on the upper slice without disturbing it.
                        let i5 = (logs.len() as f64 * 0.05).clamp(0.0, (logs.len() - 1) as f64) as usize;
                        logs.select_nth_unstable_by(i5, |a, b| a.partial_cmp(b).unwrap());
                        let i95 = (logs.len() as f64 * 0.95).clamp(0.0, (logs.len() - 1) as f64) as usize;
                        if i95 > i5 {
                            logs[i5..].select_nth_unstable_by(i95 - i5, |a, b| a.partial_cmp(b).unwrap());
                        }
                        let (mut lo, mut hi) = (logs[i5], logs[i95]);
                        if hi <= lo { hi = lo + 1.0; }
                        if lo < 0.0 { lo = 0.0; } // sub-µs floor; log10 of tiny durs still sorts fine
                        (lo, hi)
                    }
                }
            }
            None => (0.0, 1.0),
        };
        self.heat_bounds = Some(bounds);
    }

    pub(crate) fn compute_aggregates(&mut self) {
        self.sel_generation = self.sel_generation.wrapping_add(1);
        let mut all_durs: Vec<f64> = self.selection_stats.iter().flat_map(|s| s.durations.iter().copied()).collect();
        self.sel_median = median_inplace(&mut all_durs);

        self.sel_agg_stats = self.selection_stats.iter().map(|s| {
            let mut sorted = s.durations.clone();
            let median = median_inplace(&mut sorted);
            KernelStats {
                name: s.name, count: s.count, total_dur: s.total_dur,
                median_dur: median,
                max_dur: s.durations.iter().copied().fold(0.0f64, f64::max),
                min_dur: s.durations.iter().copied().fold(f64::MAX, f64::min),
            }
        }).collect();
        // First-occurrence start per aggregate row, for the chronological
        // ("Start") sort. One indexed read per selected event, same shape as
        // the totals loop above.
        let trace = self.trace.as_ref().unwrap();
        self.sel_agg_starts = self.selection_stats.iter().map(|s| {
            s.event_refs.iter().map(|&(ti, ei)| {
                trace.tracks[ti as usize].events[ei as usize].ts
            }).fold(f64::MAX, f64::min)
        }).collect();

        // One row per event, in whatever order selection_stats yields them —
        // NOT pre-sorted by duration. draw_stats_table maintains its own
        // sort_idx keyed on whichever column the user actually sorted by
        // (see ui.rs) and never assumes its input arrives sorted, so an
        // eager sort here was pure wasted work: for a 380k-event selection
        // it cost as much as the rest of this function combined, just to be
        // immediately thrown away and re-derived by sort_idx anyway.
        self.sel_individual.clear();
        self.sel_individual_refs.clear();
        for se in &self.selection_stats {
            for (i, &d) in se.durations.iter().enumerate() {
                let r = se.event_refs.get(i).copied().unwrap_or((u32::MAX, u32::MAX));
                self.sel_individual.push(KernelStats { name: se.name, count: 1, total_dur: d, median_dur: d, max_dur: d, min_dur: d });
                self.sel_individual_refs.push(r);
            }
        }
    }

}

pub struct AppState {
    pub panes: Vec<Pane>,
    pub active: usize,
    /// One-shot "force this tab selected next frame" signal, set right after
    /// opening a new trace so the tab strip visually catches up with
    /// `active`. Mirrors `Pane.pending_tab`.
    pub pending_active_tab: Option<usize>,
    pub buf: DrawBuf,
    pub bottom_h: f32,
    /// The tab strip's actual rendered width (the strip is an auto-resizing
    /// window, so this is measured, not estimated). The toolbar sizes itself
    /// against it, and a change requests the follow-up frame the auto-resize
    /// needs to converge in Wait mode.
    pub tabs_w: f32,
    pub drag: DragKind,
    pub show_diff: bool,
    pub diff_popup_open: bool,
    pub diff_result: Option<DiffResult>,
    pub diff_bar_scroll: f64,
    pub diff_bar_zoom: f64,
    pub diff_pane_indices: Option<[usize; 2]>,
    /// Target of the "right-click a kernel -> sync ranks to it" context menu,
    /// set at right-click time and read when the menu's item is clicked.
    /// `(pane_idx, event)` — pane index guards against the popup surviving a
    /// tab switch and firing against the wrong pane's trace.
    pub kernel_ctx_menu: Option<(usize, EventRef)>,
}

impl AppState {
    pub fn add_pane(&mut self) -> usize {
        self.panes.push(Pane::new());
        self.panes.len() - 1
    }

    pub fn remove_pane(&mut self, pi: usize) {
        self.panes.remove(pi);
        if self.active > pi {
            self.active -= 1;
        } else if self.active >= self.panes.len() {
            self.active = self.panes.len().saturating_sub(1);
        }
        self.drag = DragKind::None;
    }
}
