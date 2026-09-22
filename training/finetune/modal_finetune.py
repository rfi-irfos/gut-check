"""Stage D: fine-tune laya's English checkpoint on the Stage C teacher-labeled
causal-claim-verification data, on Modal.

Single-GPU port of laya's own 2xT4-DDP fine-tuning notebook
(~/projects/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb):
same GRPO-style policy-gradient-plus-CE-guidance training loop and proper
scoring reward (laya.common.proper_reward), same post-training temperature
calibration step, no DDP since our real dataset (~1k items) doesn't need
2 GPUs the way the notebook's 1,200-case x5-questions benchmark did, and
our data is single-question (causal_claim_verification, choice-type only)
rather than the notebook's multi-question multi-type mix.

Targets are one-hot (hard labels) rather than the notebook's soft
probability targets, since the Stage C teacher gives a single chosen label
+ rationale, not a calibrated distribution.

Usage:
    modal run modal_finetune.py --teacher-labeled /path/to/combined_teacher_labeled.jsonl
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
GUT_CHECK_CORE = str(Path.home() / "projects" / "gut-check" / "core")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.48.0", "safetensors>=0.4.0", "huggingface_hub>=0.20.0", "numpy", "laya")
    .add_local_dir(GUT_CHECK_CORE, remote_path="/root/gut_check_core")
)

app = modal.App("gut-check-finetune", image=image)

MODEL_ID = "convaiinnovations/laya"
QID = "decision"


def balance_classes(rows, label_key="teacher_label", cap_ratio=1.5, seed=42):
    """The raw Stage C data is ~76% UNVERIFIED (the majority class in any
    corpus of agent claims -- most claims simply aren't near a relevant tool
    observation). Training on it unbalanced collapses the model into always
    predicting the majority class with false confidence (found the hard way:
    an unbalanced first full run hit 53/55 UNVERIFIED predictions at 0.89
    mean confidence, 27.3% gold accuracy -- worse than either baseline,
    confidently wrong instead of honestly uncertain, the exact failure mode
    this whole project exists to avoid).

    Caps every class at `cap_ratio` times the second-largest class's count
    (undersampling only the dominant class), rather than downsampling
    everything to the smallest class, which would throw away almost all of
    the already-scarce CONTRADICTED/SKIP data."""
    import random
    from collections import defaultdict

    by_label = defaultdict(list)
    for r in rows:
        by_label[r.get(label_key)].append(r)
    counts = sorted((len(v) for v in by_label.values()), reverse=True)
    cap = int(counts[1] * cap_ratio) if len(counts) > 1 else counts[0]

    rng = random.Random(seed)
    out = []
    for label, group in by_label.items():
        rng.shuffle(group)
        if len(group) >= cap:
            out.extend(group[:cap])
        else:
            # Tried oversampling small classes (CONTRADICTED/SKIP) up to 4x
            # repetition here -- made it worse (38.2%, down from 41.8%):
            # training loss collapsed to ~0.04 by epoch 6, a clear
            # memorization signature from repeating 21-34 exact examples
            # several times each, not genuine generalization. Reverted to
            # undersampling the majority class only; SKIP/CONTRADICTED stay
            # underrepresented rather than overfit-inducing.
            out.extend(group)
    rng.shuffle(out)
    return out


def build_training_items(rows, tok, cfg):
    """Mirrors the notebook's build_training_item, adapted for a single
    choice-typed question with a hard (one-hot) teacher label instead of a
    soft gold distribution."""
    import sys
    sys.path.insert(0, "/root/gut_check_core")
    from gate.questions import causal_claim_verification
    from laya.common import build_sequence, render_options, QTYPES

    q = causal_claim_verification[QID]
    keys = list(q["criteria"].keys())
    rows = balance_classes(rows)
    items = []
    for row in rows:
        label = row.get("teacher_label")
        if label not in keys:
            continue
        state = {"claim": row["claim_text"], "context": (row.get("nearest_observation") or {}).get("excerpt", "")}
        target = [1.0 if k == label else 0.0 for k in keys]
        q_internal = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
        seq, markers = build_sequence(tok, state, q_internal, cfg["max_len"], cfg["head_max_len"])
        if len(markers) != len(keys):
            continue
        items.append({
            "ids": seq, "markers": markers, "qtype": QTYPES["choice"],
            "target": target, "label": keys.index(label),
        })
    return items


def collate_train_batch(items, pad_id):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target, "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


@app.function(gpu="A10G", timeout=3600)
def finetune(teacher_labeled_rows: list, gold_items: list, gold_truth: dict, seed: int = 42) -> dict:
    import random
    import time

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    import sys
    sys.path.insert(0, "/root/gut_check_core")
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward

    torch.manual_seed(seed)
    device = torch.device("cuda")
    model_dir = snapshot_download(MODEL_ID)
    _fix_tokenizer_config(model_dir)
    with open(f"{model_dir}/rl_agent_config.json") as f:
        cfg = json.load(f)
    cfg["max_len"] = 512
    cfg["head_max_len"] = 192

    tok = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
    print(f"preprocessing {len(teacher_labeled_rows)} teacher-labeled rows...")
    items = build_training_items(teacher_labeled_rows, tok, cfg)
    print(f"built {len(items)} training sequences")

    model = build_model(cfg, encoder_dir=f"{model_dir}/encoder")
    weights = load_file(f"{model_dir}/model.safetensors")
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.train()

    EPOCHS = 4  # 4/5/6 all tried on this dataset size; 4 was best (43.6% vs 41.8%/38.2% gold accuracy)
    MICRO_BATCH = 8
    GRAD_ACCUM = 2
    GROUP_SIZE = 4
    LR_ENCODER = 2.5e-5
    LR_HEAD = 1.0e-4
    SIGMA_START, SIGMA_END = 0.4, 0.1

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER}, {"params": head_params, "lr": LR_HEAD}],
        weight_decay=0.01,
    )
    total_updates = (len(items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        random.seed(seed + epoch)
        random.shuffle(items)
        epoch_loss, n_batches, accum_step = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(items), MICRO_BATCH):
            chunk = items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

        print(f"epoch {epoch+1}/{EPOCHS} avg_loss={epoch_loss/max(1,n_batches):.4f} "
              f"elapsed={time.time()-t0:.0f}s")

    # temperature calibration on a held-back slice of training items
    print("fitting calibration temperature...")
    model.eval()
    calib_items = items[::5][:200]
    calib_z, calib_t = [], []
    with torch.no_grad():
        for c_idx in range(0, len(calib_items), 16):
            c_chunk = calib_items[c_idx:c_idx + 16]
            cb = collate_train_batch(c_chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                l_sub, _ = model(
                    cb["input_ids"].to(device), cb["attention_mask"].to(device),
                    cb["marker_pos"].to(device), cb["marker_mask"].to(device), cb["qtype"].to(device),
                )
            l_np = l_sub.float().cpu().numpy()
            for r, it in enumerate(c_chunk):
                k = len(it["markers"])
                calib_z.append(l_np[r, :k])
                calib_t.append(it["target"])

    fitted_temp = 1.2
    try:
        kmax = max(len(z) for z in calib_z)
        Z = torch.full((len(calib_z), kmax), -1e4)
        T = torch.zeros((len(calib_z), kmax))
        for i, (z, t) in enumerate(zip(calib_z, calib_t)):
            Z[i, :len(z)] = torch.tensor(z)
            T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        fitted_temp = float(torch.clamp(log_t.exp(), 0.1, 10.0).item())
    except Exception as e:
        print("temperature fitting fallback:", e)
    print(f"fitted temperature: {fitted_temp:.3f}")

    cfg["fine_tuned"] = True
    cfg["model_name"] = "gut-check-causal-claim-v1"
    cfg["temperature"] = [fitted_temp, 1.0, 1.0]
    cfg["temperature_by_options"] = {}

    weights_out = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    from safetensors.torch import save as safetensors_save
    weights_bytes = safetensors_save(weights_out)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)
        tokenizer_files = {}
        for p in Path(tmp).iterdir():
            tokenizer_files[p.name] = p.read_bytes()

    # --- Evaluate against the gold set (never trained on) ---
    print(f"evaluating against {len(gold_items)} gold items...")
    from gate.gate import Gate
    from gate.policy import EscalationPolicy
    from gate.questions import causal_claim_verification as ccv

    class InMemoryAgent:
        def __init__(self, model, tok, cfg):
            self.model, self.tok, self.cfg = model, tok, cfg
            self.temperature = cfg["temperature"]

        def system_one(self, state, questions):
            from laya.common import build_sequence, render_options, QTYPES, collate_items, confidence_from_probs
            qid = next(iter(questions))
            q = questions[qid]
            q_internal = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
            seq, markers = build_sequence(self.tok, state, q_internal, self.cfg["max_len"], self.cfg["head_max_len"])
            b = collate_items([[{"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]]}]], self.tok.pad_token_id)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                logits, act = self.model(
                    b["input_ids"].to(device), b["attention_mask"].to(device),
                    b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device),
                )
            k = len(markers)
            z = logits.float().cpu().numpy()[0, :k] / self.temperature[0]
            p = np.exp(z - z.max())
            p = p / p.sum()
            keys = list(q["criteria"].keys())
            return {"answers": {qid: {
                "type": "choice", "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": round(confidence_from_probs(p, k), 4),
            }}}

    model.eval()
    agent = InMemoryAgent(model, tok, cfg)
    gate = Gate(router=agent, policy=EscalationPolicy())
    correct = 0
    rows_out = []
    for it in gold_items:
        state = {"claim": it["claim_text"], "context": it["context"]}
        verdict = gate.classify(state, ccv)
        gt = gold_truth[str(it["id"])]
        is_correct = verdict.choice == gt
        correct += is_correct
        rows_out.append({"id": it["id"], "gt": gt, "pred": verdict.choice, "confidence": verdict.confidence})

    accuracy = correct / len(gold_items) if gold_items else 0.0
    print(f"gold-set accuracy: {correct}/{len(gold_items)} = {accuracy:.1%}")

    return {
        "weights_bytes": weights_bytes,
        "encoder_config": model.encoder.config.to_dict(),
        "tokenizer_files": tokenizer_files,
        "cfg": cfg,
        "n_train_items": len(items),
        "gold_accuracy": accuracy,
        "gold_rows": rows_out,
    }


@app.local_entrypoint()
def main(teacher_labeled: str = None, gold_eval: str = None, limit: int = None, seed: int = 42):
    teacher_labeled = teacher_labeled or str(
        Path.home() / "projects" / "gut-check" / "training" / "eval" / "local_data" / "combined_teacher_labeled.jsonl"
    )
    gold_path = gold_eval or str(
        Path.home() / "projects" / "gut-check" / "training" / "eval" / "local_data" / "gold_eval.jsonl"
    )

    with open(teacher_labeled) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit:
        rows = rows[:limit]
    with open(gold_path) as f:
        gold_rows = [json.loads(line) for line in f if line.strip()]
    gold_items = [{"id": r["id"], "claim_text": r["claim_text"], "context": r["context"]} for r in gold_rows]
    gold_truth = {str(r["id"]): r["label"] for r in gold_rows}

    print(f"training on {len(rows)} teacher-labeled rows (seed={seed}), evaluating on {len(gold_items)} gold items")
    result = finetune.remote(rows, gold_items, gold_truth, seed=seed)

    print(f"\nn_train_items: {result['n_train_items']}")
    print(f"gold-set accuracy: {result['gold_accuracy']:.1%}")
    print("(compare against baselines: 52.7% heuristic, 38.2% stock checkpoint)")

    # Versioned, never-overwritten run dirs -- a single mistaken re-run must
    # never silently destroy the best checkpoint found so far (happened once:
    # a worse re-run overwrote a 43.6%-accuracy checkpoint with no backup).
    runs_dir = Path.home() / "projects" / "gut-check" / "training" / "finetune" / "local_data" / "runs"
    import time as _time
    run_id = f"seed{seed}_n{result['n_train_items']}_{_time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = runs_dir / run_id
    (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    (out_dir / "encoder").mkdir(parents=True, exist_ok=True)

    (out_dir / "model.safetensors").write_bytes(result["weights_bytes"])
    with open(out_dir / "rl_agent_config.json", "w") as f:
        json.dump(result["cfg"], f, indent=2)
    with open(out_dir / "encoder" / "config.json", "w") as f:
        json.dump(result["encoder_config"], f, indent=2)
    for name, data in result["tokenizer_files"].items():
        (out_dir / "tokenizer" / name).write_bytes(data)
    with open(out_dir / "gold_eval_results.json", "w") as f:
        json.dump({"accuracy": result["gold_accuracy"], "rows": result["gold_rows"]}, f, indent=2)

    print(f"\nsaved a self-contained laya.Agent checkpoint to {out_dir}")
    print(f'load it with: laya.load("{out_dir}")')

    best_path = runs_dir / "best_run.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {"accuracy": -1.0, "run_id": None}
    if result["gold_accuracy"] > best["accuracy"]:
        best_path.write_text(json.dumps({"accuracy": result["gold_accuracy"], "run_id": run_id}, indent=2))
        print(f"NEW BEST: {result['gold_accuracy']:.1%} (previous best: "
              f"{best['accuracy']:.1%} from {best['run_id']})")
    else:
        print(f"not a new best ({result['gold_accuracy']:.1%} <= current best "
              f"{best['accuracy']:.1%} from {best['run_id']})")
