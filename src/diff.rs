use crate::types::*;

const WINDOW: usize = 200;

pub fn compute_diff(seq_a: &[(String, f64)], seq_b: &[(String, f64)]) -> DiffResult {
    let mut lines = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < seq_a.len() && j < seq_b.len() {
        if seq_a[i].0 == seq_b[j].0 {
            lines.push(DiffLine {
                kind: DiffKind::Same,
                name: seq_a[i].0.clone(),
                dur_a: Some(seq_a[i].1),
                dur_b: Some(seq_b[j].1),
            });
            i += 1;
            j += 1;
            continue;
        }

        let limit_b = seq_b.len().min(j + WINDOW);
        let limit_a = seq_a.len().min(i + WINDOW);
        let found_in_b = (j + 1..limit_b).find(|&k| seq_b[k].0 == seq_a[i].0);
        let found_in_a = (i + 1..limit_a).find(|&k| seq_a[k].0 == seq_b[j].0);

        match (found_in_b, found_in_a) {
            (Some(bk), Some(ak)) => {
                let cost_b = bk - j;
                let cost_a = ak - i;
                if cost_b <= cost_a {
                    for k in j..bk {
                        lines.push(DiffLine {
                            kind: DiffKind::Added,
                            name: seq_b[k].0.clone(),
                            dur_a: None,
                            dur_b: Some(seq_b[k].1),
                        });
                    }
                    j = bk;
                } else {
                    for k in i..ak {
                        lines.push(DiffLine {
                            kind: DiffKind::Removed,
                            name: seq_a[k].0.clone(),
                            dur_a: Some(seq_a[k].1),
                            dur_b: None,
                        });
                    }
                    i = ak;
                }
            }
            (Some(bk), None) => {
                for k in j..bk {
                    lines.push(DiffLine {
                        kind: DiffKind::Added,
                        name: seq_b[k].0.clone(),
                        dur_a: None,
                        dur_b: Some(seq_b[k].1),
                    });
                }
                j = bk;
            }
            (None, Some(ak)) => {
                for k in i..ak {
                    lines.push(DiffLine {
                        kind: DiffKind::Removed,
                        name: seq_a[k].0.clone(),
                        dur_a: Some(seq_a[k].1),
                        dur_b: None,
                    });
                }
                i = ak;
            }
            (None, None) => {
                lines.push(DiffLine {
                    kind: DiffKind::Removed,
                    name: seq_a[i].0.clone(),
                    dur_a: Some(seq_a[i].1),
                    dur_b: None,
                });
                lines.push(DiffLine {
                    kind: DiffKind::Added,
                    name: seq_b[j].0.clone(),
                    dur_a: None,
                    dur_b: Some(seq_b[j].1),
                });
                i += 1;
                j += 1;
            }
        }
    }

    while i < seq_a.len() {
        lines.push(DiffLine {
            kind: DiffKind::Removed,
            name: seq_a[i].0.clone(),
            dur_a: Some(seq_a[i].1),
            dur_b: None,
        });
        i += 1;
    }
    while j < seq_b.len() {
        lines.push(DiffLine {
            kind: DiffKind::Added,
            name: seq_b[j].0.clone(),
            dur_a: None,
            dur_b: Some(seq_b[j].1),
        });
        j += 1;
    }

    let total_dur_a: f64 = seq_a.iter().map(|(_, d)| d).sum();
    let total_dur_b: f64 = seq_b.iter().map(|(_, d)| d).sum();

    DiffResult {
        lines,
        count_a: seq_a.len() as u32,
        count_b: seq_b.len() as u32,
        total_dur_a,
        total_dur_b,
    }
}
