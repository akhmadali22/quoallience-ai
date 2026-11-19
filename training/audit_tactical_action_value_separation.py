#!/usr/bin/env python3
"""Inference-only audit of tactical action preferences and value separation."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from statistics import mean, median, pstdev
from quoridor.engine import GameState, apply_action_index, validate_state
from quoridor.neural_ai.encoder import ACTION_SIZE, decode_action, encode_state, legal_action_indices
from quoridor.neural_ai.policy import load_model
from training.audit_tactical_label_head_calibration import independent_immediate_win, restore_state, sample_controls

CLASSES = ("immediate_win", "prevents_immediate_loss", "allows_immediate_loss",
           "creates_immediate_loss", "no_immediate_terminal_consequence")

def describe_action(index):
    kind, payload = decode_action(index)
    if kind == "move": return {"type":"pawn","to":list(payload)}
    return {"type":"wall","orientation":payload.orientation,"x":payload.x,"y":payload.y}

def classify_action(state, action, pre_threat):
    child = apply_action_index(state, action); root = state.current_player
    if child.winner is not None:
        return ("immediate_win" if child.winner == root else "allows_immediate_loss", 1.0 if child.winner == root else -1.0)
    post_threat = independent_immediate_win(child, 1-root)
    if pre_threat and not post_threat: category = "prevents_immediate_loss"
    elif pre_threat and post_threat: category = "allows_immediate_loss"
    elif not pre_threat and post_threat: category = "creates_immediate_loss"
    else: category = "no_immediate_terminal_consequence"
    return category, float("nan")

def distribution(values):
    if not values: return {"count":0,"mean":None,"median":None,"std":None,"min":None,"p10":None,"p25":None,"p75":None,"p90":None,"max":None}
    s=sorted(values)
    def q(p):
        if len(s)==1: return s[0]
        pos=(len(s)-1)*p; lo=math.floor(pos); hi=math.ceil(pos)
        return s[lo]+(s[hi]-s[lo])*(pos-lo)
    return {"count":len(s),"mean":mean(s),"median":median(s),"std":pstdev(s),"min":s[0],
            "p10":q(.1),"p25":q(.25),"p75":q(.75),"p90":q(.9),"max":s[-1]}

def pairwise_auc(rows, positive, negative, score_key):
    groups={}
    for r in rows: groups.setdefault(r["position_id"],{}).setdefault(r["class"],[]).append(r[score_key])
    wins=ties=pairs=0
    for g in groups.values():
        for a in g.get(positive,[]):
            for b in g.get(negative,[]):
                pairs+=1
                if a>b: wins+=1
                elif a==b: ties+=1
    return (wins+.5*ties)/pairs if pairs else None

def evaluate(model,state,cohort,position_id,device,batch_size):
    import torch
    validate_state(state); legal=legal_action_indices(state)
    if not legal: raise RuntimeError("Non-terminal audit position has no legal actions")
    root=state.current_player; pre_threat=independent_immediate_win(state,1-root)
    x=torch.tensor(encode_state(state),dtype=torch.float32,device=device).unsqueeze(0)
    with torch.inference_mode(): logits,root_value=model(x)
    masked=torch.full((ACTION_SIZE,),float("-inf"),device=device); masked[legal]=logits[0][legal]
    with torch.inference_mode(): probs=torch.softmax(masked,dim=0)
    priors={a:float(probs[a].item()) for a in legal}
    entropy=-sum(p*math.log(max(p,1e-30)) for p in priors.values() if p>0)
    norm_entropy=entropy/math.log(len(legal)) if len(legal)>1 else 0.0
    rows=[]; children=[]; child_actions=[]; child_values={}
    for a in legal:
        child=apply_action_index(state,a); category,terminal_value=classify_action(state,a,pre_threat)
        if child.winner is not None: child_values[a]=terminal_value
        else: children.append(child); child_actions.append(a)
        rows.append({"action_index":a,"action":describe_action(a),"class":category,"policy_prior":priors[a]})
    for start in range(0,len(children),batch_size):
        batch=torch.tensor([encode_state(s) for s in children[start:start+batch_size]],dtype=torch.float32,device=device)
        with torch.inference_mode(): _,vals=model(batch)
        for a,v in zip(child_actions[start:start+batch_size],vals.detach().cpu().tolist()): child_values[a]=-float(v)
    po=sorted(legal,key=lambda a:(-priors[a],a)); vo=sorted(legal,key=lambda a:(-child_values[a],a))
    pr={a:i for i,a in enumerate(po,1)}; vr={a:i for i,a in enumerate(vo,1)}
    for r in rows:
        a=r["action_index"]; r.update(position_id=position_id,value_root_perspective=child_values[a],policy_rank=pr[a],value_rank=vr[a])
    stats={}
    for c in CLASSES:
        selected=[r for r in rows if r["class"]==c]
        stats[c]={"action_count":len(selected),"policy_mass":sum(r["policy_prior"] for r in selected),
                  "mean_action_prior":mean(r["policy_prior"] for r in selected) if selected else None,
                  "mean_child_value":mean(r["value_root_perspective"] for r in selected) if selected else None,
                  "median_child_value":median(r["value_root_perspective"] for r in selected) if selected else None,
                  "best_policy_rank":min((r["policy_rank"] for r in selected),default=None),
                  "best_value_rank":min((r["value_rank"] for r in selected),default=None)}
    pawn=sum(priors[a] for a in legal if decode_action(a)[0]=="move")
    return {"position_id":position_id,"cohort":cohort,"state":state.to_dict(),"ply":state.ply,
            "wall_count":len(state.walls),"legal_action_count":len(legal),"opponent_immediate_win_before_action":pre_threat,
            "root_value_head":float(root_value[0].item()),"policy_entropy":entropy,
            "policy_entropy_normalized":norm_entropy,"pawn_policy_mass":pawn,"wall_policy_mass":1-pawn,
            "action_class_summary":stats,"actions":rows}

def summarize_positions(positions):
    if not positions:return {"position_count":0}
    actions=[a for p in positions for a in p["actions"]]; classes={}
    for c in CLASSES:
        selected=[a for a in actions if a["class"]==c]
        classes[c]={"action_count":len(selected),"position_count":len({a["position_id"] for a in selected}),
          "policy_mass_per_position":distribution([p["action_class_summary"][c]["policy_mass"] for p in positions]),
          "action_prior":distribution([a["policy_prior"] for a in selected]),
          "child_value_root_perspective":distribution([a["value_root_perspective"] for a in selected]),
          "policy_rank":distribution([float(a["policy_rank"]) for a in selected]),
          "value_rank":distribution([float(a["value_rank"]) for a in selected])}
    return {"position_count":len(positions),"action_row_count":len(actions),
      "root_value_head":distribution([p["root_value_head"] for p in positions]),
      "policy_entropy":distribution([p["policy_entropy"] for p in positions]),
      "policy_entropy_normalized":distribution([p["policy_entropy_normalized"] for p in positions]),
      "pawn_policy_mass":distribution([p["pawn_policy_mass"] for p in positions]),
      "wall_policy_mass":distribution([p["wall_policy_mass"] for p in positions]),
      "class_summary":classes,
      "within_position_pairwise_auc":{
       "policy_prevents_vs_allows":pairwise_auc(actions,"prevents_immediate_loss","allows_immediate_loss","policy_prior"),
       "value_prevents_vs_allows":pairwise_auc(actions,"prevents_immediate_loss","allows_immediate_loss","value_root_perspective"),
       "policy_immediate_win_vs_quiet":pairwise_auc(actions,"immediate_win","no_immediate_terminal_consequence","policy_prior"),
       "value_immediate_win_vs_quiet":pairwise_auc(actions,"immediate_win","no_immediate_terminal_consequence","value_root_perspective")}}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("checkpoint",type=Path)
    p.add_argument("--source-json",type=Path,default=Path("data/diagnostics/tactical_policy_value_attribution/run_001/tactical_policy_value_attribution.json"))
    p.add_argument("--control-positions",type=int,default=30); p.add_argument("--seed",type=int,default=20261121)
    p.add_argument("--max-games",type=int,default=5000); p.add_argument("--max-plies",type=int,default=160)
    p.add_argument("--batch-size",type=int,default=128); p.add_argument("--device",default="cpu"); p.add_argument("--threads",type=int,default=1)
    p.add_argument("--output-dir",type=Path,default=Path("data/diagnostics/tactical_action_value_separation/run_001"))
    a=p.parse_args()
    if not a.checkpoint.is_file() or not a.source_json.is_file(): p.error("Checkpoint and --source-json must exist")
    if min(a.control_positions,a.batch_size,a.threads)<1:p.error("control-positions, batch-size, threads must be positive")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):p.error(f"Output directory is not empty: {a.output_dir}")
    import torch
    torch.set_num_threads(a.threads)
    if a.device.startswith("cuda") and not torch.cuda.is_available():p.error("CUDA requested but unavailable")
    source=json.loads(a.source_json.read_text(encoding="utf-8")); frozen=source.get("positions",[])
    if not frozen:p.error("Source JSON contains no positions[]")
    model=load_model(a.checkpoint,device=a.device); model.eval()
    sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(); positions=[]
    for i,item in enumerate(frozen):
        s=restore_state(item["state"])
        positions.append(evaluate(model,s,"frozen_threat_with_defense",item.get("position_id",f"frozen_{i:03d}"),a.device,a.batch_size))
    controls=sample_controls(a.seed,a.control_positions,a.max_games,a.max_plies)
    for cohort,states in controls.items():
        for i,s in enumerate(states):positions.append(evaluate(model,s,cohort,f"{cohort}_{i:03d}",a.device,a.batch_size))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    cohorts={n:summarize_positions([p for p in positions if p["cohort"]==n]) for n in sorted({p["cohort"] for p in positions})}
    report={"experiment":"tactical_action_preference_and_value_separation","checkpoint":str(a.checkpoint),
      "checkpoint_sha256":sha,"source_json_sha256":hashlib.sha256(a.source_json.read_bytes()).hexdigest(),
      "sampler":{"frozen_positions":len(frozen),"control_positions_per_cohort":a.control_positions,"control_seed":a.seed,
                 "control_method":"seeded legal playout; reused validated control sampler"},
      "evaluation":{"mcts_or_puct_used":False,"checkpoint_modified":False,"training_performed":False,
       "policy":"legal-action masked softmax over raw logits",
       "value":"child value converted to root-player perspective; terminal children exact +/-1",
       "classification":{"immediate_win":"root action wins immediately",
        "prevents_immediate_loss":"opponent has immediate win before action, none after",
        "allows_immediate_loss":"opponent has immediate win both before and after action",
        "creates_immediate_loss":"no immediate opponent win before action, one exists after",
        "no_immediate_terminal_consequence":"nonterminal child and no immediate opponent winning reply"}},
      "cohort_summary":cohorts,"positions":positions}
    (a.output_dir/"tactical_action_value_separation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (a.output_dir/"summary.json").write_text(json.dumps({k:v for k,v in report.items() if k!="positions"},indent=2)+"\n",encoding="utf-8")
    print(f"Positions: {len(positions)}; action rows: {sum(len(p['actions']) for p in positions)}")
    print(f"Checkpoint SHA-256: {sha}"); print(f"Summary: {a.output_dir/'summary.json'}")
    return 0
if __name__=="__main__": raise SystemExit(main())
