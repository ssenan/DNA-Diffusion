import glob
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dnadiffusion import DATA_DIR
from dnadiffusion.utils.data_util import seq_extract, write_fasta_blat


def run_blat(df_file_path: str, tag_list: List, download_blat_script: bool = False) -> None:
    """Downloads blat script from UCSC and runs blat on the train, test and generated sequences

    Args:
        df_file_path (str): Path to the dataframe file
        tag_list (List): List of tags to extract from the dataframe and run blat on
        download_blat_script (bool, optional): Whether to download the blat script from UCSC. Defaults to False.

    Returns:
        None
    """
    if download_blat_script:
        # Downloading blat script from UCSC
        os.system(
            f"wget 'http://genome-source.soe.ucsc.edu/gitlist/kent.git/raw/master/src/utils/pslScore/pslScore.pl' -O pslScore.pl"
        )

    # Creating fasta files for blast
    for tag in tag_list:
        df = seq_extract(df_file_path, tag)
        write_fasta_blat(df, f"{DATA_DIR}/{tag}.fasta")

    # Running blat with 90 percent identity
    # train vs generated sequences
    os.system(
        f"blat -t=dna -q=dna -tileSize=11 -stepSize=5 -minMatch=1 -repMatch=2253  -minScore=20  -minIdentity=90 -noHead -out=psl  {DATA_DIR}/GENERATED.fa {DATA_DIR}/train.fa   {DATA_DIR}/output_train_generated_90minseqid.psl"
    )
    # train vs test sequences
    os.system(
        f"blat -t=dna -q=dna -tileSize=11 -stepSize=5 -minMatch=1 -repMatch=2253  -minScore=20  -minIdentity=90 -noHead -out=psl  {DATA_DIR}/test.fa {DATA_DIR}/train.fa   {DATA_DIR}/output_train_test_90minseqid.psl"
    )
    # train vs random sequences
    os.system(
        f"blat -t=dna -q=dna -tileSize=11 -stepSize=5 -minMatch=1 -repMatch=2253  -minScore=20  -minIdentity=90 -noHead -out=psl  {DATA_DIR}/RANDOM_GENOME_REGIONS.fa {DATA_DIR}/train.fa   {DATA_DIR}/output_train_random_90minseqid.psl"
    )

    # Running blat with 100 percent identity
    # train vs generated sequences
    os.system(
        f"blat -t=dna -q=dna -tileSize=11 -stepSize=5 -minMatch=1 -repMatch=2253  -minScore=20  -minIdentity=100 -noHead -out=psl  {DATA_DIR}/GENERATED.fa {DATA_DIR}/train.fa   {DATA_DIR}/output_train_generated.psl"
    )
    # train vs test sequences
    os.system(
        f"blat -t=dna -q=dna -tileSize=11 -stepSize=5 -minMatch=1 -repMatch=2253  -minScore=20  -minIdentity=100 -noHead -out=psl  {DATA_DIR}/test.fa {DATA_DIR}/train.fa   {DATA_DIR}/output_train_test.psl"
    )
    # train vs random sequences
    os.system(
        f"blat -t=dna -q=dna -tileSize=11 -stepSize=5 -minMatch=1 -repMatch=2253  -minScore=20  -minIdentity=100 -noHead -out=psl  {DATA_DIR}/RANDOM_GENOME_REGIONS.fa {DATA_DIR}/train.fa   {DATA_DIR}/output_train_random.psl"
    )

    # call pslScore.pl to get the scores for all the alignments *psl files
    psl_files = glob.glob(f"{DATA_DIR}/*.psl")

    for psl_file in psl_files:
        output_file = f"score_{psl_file.replace('.psl', '')}.txt"
        cmd = f"perl pslScore.pl {psl_file} > {output_file}"
        os.system(cmd)

    print("blat complete")


## util function to manage PSL files (BLAT output)
def split_score_hits_from_psl(
    psl_score_file: str, return_header: bool = False, invert_hits_and_queries: bool = False
) -> pd.DataFrame:
    """Function to split the hits into a query-like range on the output provided by the pslScore.pl script once it has been run on a PSL file

    Args:
        psl_score_file: Path to the output file from pslScore.pl
        invert_hits_and_queries: Boolean to indicate whether to invert the hit and query column ranges in the output DataFrame.
    Returns:
        split_df: A DataFrame containing the split hits
    """
    _df = pd.read_table(psl_score_file, sep="\t", header=None)
    _df[[3, "hit_range"]] = _df[3].str.split(":", expand=True)
    _df[["hit_range_start", "hit_range_end"]] = _df["hit_range"].str.split("-", expand=True)
    _df = _df.drop(columns=["hit_range"])
    _df.rename(
        columns={0: "query_id", 1: "query_range_start", 2: "query_range_end", 3: "hit_id", 4: "bp_length", 5: "score"},
        inplace=True,
    )

    if invert_hits_and_queries:
        column_names = [
            "hit_id",
            "hit_range_start",
            "hit_range_end",
            "bp_length",
            "query_id",
            "query_range_start",
            "query_range_end",
            "score",
        ]
    else:
        column_names = [
            "query_id",
            "query_range_start",
            "query_range_end",
            "hit_id",
            "hit_range_start",
            "hit_range_end",
            "bp_length",
            "score",
        ]
    split_df = _df.reindex(columns=column_names)
    return split_df


def blat_plot_preprocess(input_file: str, df_file_path: str, tag: str) -> pd.DataFrame:
    """Preprocess the output of blat to be used for plotting

    Args:
        input_file: Path to the output file from pslScore.pl
        df_file_path: Path to the dataframe file
        tag: Tag to extract from the dataframe

    Returns:
        merged_df: A DataFrame containing the merged hits
    """
    min_seq_id = split_score_hits_from_psl(input_file)
    # subset on bp_length as well to keep highest value
    min_seq_id = min_seq_id.sort_values(["query_id", "bp_length"]).drop_duplicates(subset=["query_id"], keep="last")
    # remove unused columns
    min_seq_id = min_seq_id.drop(
        ["query_range_start", "query_range_end", "hit_id", "hit_range_start", "hit_range_end", "score"], axis=1
    )
    # rename columns for join
    min_seq_id.rename(columns={"query_id": "ID"}, inplace=True)
    df = seq_extract(df_file_path, tag)
    df = df.drop(["chrom", "start", "end", "CELL_TYPE", "TAG"], axis=1)
    merged_df = pd.merge(df, min_seq_id, on="ID", how="left")
    merged_df = merged_df.assign(Dataset=f"Train x {tag}")
    return merged_df


def create_blat_plot_df(df_file_path: str, tag_list: List) -> pd.DataFrame:
    """Create a DataFrame containing the merged hits for all the tags

    Args:
        df_file_path: Path to the dataframe file
        tag_list: List of tags to extract from the dataframe and run blat on

    Returns:
        merged_df: A DataFrame containing the merged hits across 90 and 100 percent sequence identity
    """
    merged_df_list_90 = []
    merged_df_list_100 = []
    for tag in tag_list:
        input_file_90 = f"{DATA_DIR}/score_output_train_{tag}_90minseqid.txt"
        merged_df_list_90.append(blat_plot_preprocess(input_file_90, df_file_path, tag))
        input_file_100 = f"{DATA_DIR}/score_output_train_{tag}.txt"
        merged_df_list_100.append(blat_plot_preprocess(input_file_100, df_file_path, tag))

    merged_df_90 = pd.melt(pd.concat(merged_df_list_90), id_vars=["Dataset"], value_vars=["bp_length"])
    merged_df_100 = pd.melt(pd.concat(merged_df_list_100), id_vars=["Dataset"], value_vars=["bp_length"])

    merged_df = pd.merge(merged_df_90, merged_df_100[["ID", "bp_length"]], on="ID", how="left")
    merged_df.rename(
        columns={"bp_length_x": "90_seq_id_bp_length", "bp_length_y": "100_seq_id_bp_length"}, inplace=True
    )
    merged_df = merged_df[["ID", "SEQUENCE", "90_seq_id_bp_length", "100_seq_id_bp_length", "Dataset"]]
    return merged_df


def create_blat_boxplot(
    input_df: pd.DataFrame, value_vars: str, output_file: str, id_var_column: str = "Dataset", nan_fill: bool = False,
) -> None:
    """Create a boxplot from the input DataFrame

    Args:
        input_df (pd.DataFrame): Input DataFrame containing the merged hits across 90 and 100 percent sequence identity
        id_var_column (str): Column name to use as the ID variable. Default is "Dataset" 
        value_vars (str): variables that are unpivotted into the boxplot. Possible options are ["90_seq_id_bp_length", "100_seq_id_bp_length"]
        output_file (str): Path to save the output plot

    Returns:
        None
    """
    # Resetting plot
    plt.clf()
    # Creating boxplot
    df_plot = pd.melt(input_df, id_vars=[id_var_column], value_vars=value_vars)
    if nan_fill:
        df_plot = df_plot.fillna(value={"value": 19})
    ax = sns.boxplot(data=df_plot, x="Dataset", y="value", showfliers=True, hue="Dataset")
    ax.set(ylabel="Alignment length/bp", title="Alignments by maximum bp value | 90 min_seq_id")
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":

    pass