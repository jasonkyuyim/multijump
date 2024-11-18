import pandas as pd
from pathlib import Path
import numpy as np
import os
import shutil
from tqdm import tqdm
from multiflow.models import utils as mu

db_path = '/data/rbg/users/jyim/third_party/pdb'

def foldseek_command(sample_dir):
    aln_path = os.path.join(os.path.dirname(sample_dir), "aln.m8")
    foldseek_args = [
        "foldseek",
        "easy-search",
        sample_dir,
        db_path,
        aln_path,
        "tmpFolder",
        "--alignment-type",
        "1",
        "--format-output",
        "query,target,alntmscore,lddt",
        "--tmscore-threshold",
        "0.0",
        "--exhaustive-search",
        "--max-seqs",
        "10000000000",
        "--comp-bias-corr",
        "0",
        "--mask",
        "0",
    ]
    return " ".join(foldseek_args), aln_path

def calc_novelty(aln_path):
    foldseek_df = {
        'sample': [],
        'alntm': [],
    }
    with open(aln_path) as file:
        for item in file:
            file, _, _, tm_score = item.split('\t')
            tm_score = float(tm_score)
            foldseek_df['sample'].append(file)
            foldseek_df['alntm'].append(tm_score)
    foldseek_df = pd.DataFrame(foldseek_df)
    novelty_summary = foldseek_df.groupby('sample').agg({'alntm': 'max'}).reset_index()
    return novelty_summary

def read_results(path_list):
    df_list = []
    for path in path_list:
        df = pd.read_csv(path)
        sample_dir = path.parent / "designable/*.pdb"
        novelty_cmd, aln_path = foldseek_command(str(sample_dir))
        print(novelty_cmd)
        df["aln_path"] = aln_path
        df["path"] = path
        df_list.append(df)
    return pd.concat(df_list)


def read_pmpnn_8_results(path_list):
    df_list = []
    for path in path_list:
        df = pd.read_csv(path)
        sample_dir = path.parent / "designable/*.pdb"
        novelty_cmd, aln_path = foldseek_command(str(sample_dir))
        print(novelty_cmd)
        df["aln_path"] = aln_path
        df["path"] = path
        df_list.append(df)
    return pd.concat(df_list)

def process(results_dir):
    designable_csv_path = results_dir / "pmpnn_designable.csv"
    designable_df = {}
    pmpnn_designable_dir = results_dir / "pmpnn_designable"
    (results_dir / "pmpnn_designable").mkdir(exist_ok=True)
    designable_text_path = pmpnn_designable_dir / "designable.txt"
    designable = 0
    total = 0
    print(f"Processing with {results_dir}")
    with open(designable_text_path, "w") as file:
        for pmpn_csv_path in tqdm(results_dir.glob("length_*/sample_*/pmpnn_results.csv")):
            length = str(pmpn_csv_path.parents[1].stem.split("_")[1])
            sample_id = str(pmpn_csv_path.parents[0].stem.split("_")[1])
            pmpnn_df = pd.read_csv(pmpn_csv_path).sort_values('bb_rmsd', ascending=True)
            best_row = pmpnn_df.iloc[0]
            total += 1
            if best_row.bb_rmsd < 2.0:
                designable += 1
                file.write(f"{best_row.folded_path}\n")
                shutil.copy(best_row.sample_path, pmpnn_designable_dir / f"len_{length}_id_{sample_id}.pdb")
    novelty_cmd, aln_path = foldseek_command(str(pmpnn_designable_dir))
    print(novelty_cmd)
    clusters = mu.run_max_cluster(designable_text_path, pmpnn_designable_dir)
    designable_df["designable"] = [designable / total]
    designable_df["diversity"] = [clusters]
    designable_df["aln_path"] = [aln_path]
    designable_df["path"] = [pmpnn_designable_dir]
    pd.DataFrame(designable_df).to_csv(designable_csv_path)
    print(f"Done with {results_dir}")

def run():
    base_dir = Path("/data/rbg/users/jyim/projects/multiflow/")

    all_dirs = [
        # base_dir / "inference_outputs/weights/last/unconditional/run_2024-11-14_18-31-45",
        # base_dir / "inference_outputs/weights/last/unconditional/run_jumps_0.0_flow_1.0_2024-11-15_17-02-56",
        # base_dir / "inference_outputs/weights/last/unconditional/run_jumps_0.0_flow_1.0_2024-11-15_22-41-03",

        # base_dir / 'inference_outputs/weights/last/unconditional/run_jumps_2024-11-14_21-37-39',
        # base_dir / 'inference_outputs/weights/last/unconditional/run_jumps_1.0_flow_0.0_2024-11-15_09-53-28',
        # base_dir / 'inference_outputs/weights/last/unconditional/run_jumps_1.0_flow_0.0_2024-11-15_21-43-30',

        # base_dir / "inference_outputs/weights/last/unconditional/run_jumps_2024-11-14_21-40-39",
        # base_dir / "inference_outputs/weights/last/unconditional/run_jumps_0.5_flow_0.5_2024-11-15_10-08-39",
        # base_dir / "inference_outputs/weights/last/unconditional/run_jumps_0.5_flow_0.5_2024-11-16_10-16-07",

        base_dir / "inference_outputs/weights/last/unconditional/run_jumps_0.5_flow_0.5_2024-11-15_10-11-51",
        base_dir / "inference_outputs/weights/last/unconditional/run_jumps_0.5_flow_0.5_2024-11-15_21-41-31",
        base_dir / "inference_outputs/weights/last/unconditional/run_jumps_0.5_flow_0.5_2024-11-16_10-14-07",
    ]
    for results_dir in all_dirs:
        process(results_dir)
    


if __name__ == "__main__":
    run()