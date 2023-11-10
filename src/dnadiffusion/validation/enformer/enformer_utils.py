import configparser
import gzip
import json

import kipoiseq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import seaborn as sns
from kipoiseq import Interval

from dnadiffusion import DATA_DIR
from dnadiffusion.validation.enformer.enformer import FastaStringExtractor
from dnadiffusion.validation.validation_utils import quantile_normalization


def variant_generator(vcf_file, gzipped=False):
    """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""

    def _open(file):
        return gzip.open(vcf_file, "rt") if gzipped else open(vcf_file)

    with _open(vcf_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom, pos, id, ref, alt_list = line.split("\t")[:5]
            # Split ALT alleles and return individual variants as output.
            for alt in alt_list.split(","):
                yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos, ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False, chr_prefix=""):
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=FastaStringExtractor(fasta_file))

    for variant in variant_generator(vcf_file, gzipped=gzipped):
        interval = Interval(chr_prefix + variant.chrom, variant.pos, variant.pos)
        interval = interval.resize(sequence_length)
        center = interval.center() - interval.start

        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)

        yield {
            "inputs": {"ref": one_hot_encode(reference), "alt": one_hot_encode(alternate)},
            "metadata": {
                "chrom": chr_prefix + variant.chrom,
                "pos": variant.pos,
                "id": variant.id,
                "ref": variant.ref,
                "alt": variant.alt,
            },
        }


def plot_tracks(tracks, interval, height=1.5, color="blue", set_y=False):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y, color=color)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    # plt.tight_layout()
    if set_y:
        plt.ylim(set_y[0], set_y[1])


def normalize_tracks(df_to_normalize: pd.DataFrame, normalize_df_helper: pd.DataFrame, boxplot: bool = False):
    tranpose_data = pd.concat([df_to_normalize.T, normalize_df_helper.T], axis=1)
    df_normalized = pd.DataFrame(quantile_normalization(tranpose_data.values.T))
    df_normalized.columns = tranpose_data.index
    output_tracks = df_normalized.head(df_to_normalize.shape[0])

    if boxplot:
        plt.clf()
        sns.boxplot(x="variable", y="value", data=output_tracks.melt())
        # Save the plot
        plt.savefig(f"{DATA_DIR}/boxplot.png", dpi=300)

    return output_tracks


def extract_value_json(
    json_path: str,
    big_wig_files: list,
    gene_region: list,
    gene_region_offset: int = 1000,
):
    # Collect file names 3 at a time
    max_dict = {}
    for i in range(0, len(sorted(big_wig_files)), 3):
        curr_files = big_wig_files[i : i + 3]
        # For each group of 3 files, collect the max value for each file
        curr_max = []
        for file in curr_files:
            bw = pyBigWig.open(file)
            curr_max.append(
                bw.stats(
                    gene_region[0], gene_region[1] - gene_region_offset, gene_region[2] + gene_region_offset, type="max"
                )[0]
            )
            bw.close()
        # add the max values to the dictionary for each individual file
        for file in curr_files:
            # replace extension with .bedgraph
            file_rename = file.replace(".bigwig", ".bedgraph")
            max_dict[file_rename] = max(curr_max) * 1.5

    with open(json_path) as json_file:
        data = json.load(json_file)

    # Match the max values to the correct file in the json
    for track in data:
        if track["url"] in max_dict.keys():
            # Using regex replace each instance of <"value"> with the max value
            track["max"] = track["max"].replace("<value>", str(max_dict[track["url"]]))

    # Writing the updated json file
    output_name = "_".join([str(i) for i in gene_region])
    with open(f"{DATA_DIR}/{output_name}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    return output_name + ".json"


def extract_value_ini(
    ini_path: str,
    big_wig_files: list,
    enhancer_region: list,
    enhancer_region_offset: int = 1000,
    custom_max_dict: dict | None = None,
):
    # Collect file names 3 at a time
    max_dict = {}
    if not custom_max_dict:
        for i in range(0, len(sorted(big_wig_files)), 3):
            curr_files = big_wig_files[i : i + 3]
            # For each group of 3 files, collect the max value for each file
            curr_max = []
            for file in curr_files:
                bw = pyBigWig.open(file)
                curr_max.append(
                    bw.stats(
                        enhancer_region[0],
                        enhancer_region[1] - enhancer_region_offset,
                        enhancer_region[2] + enhancer_region_offset,
                        type="max",
                    )[0]
                )
                bw.close()
            # add the max values to the dictionary for each individual file
            for file in curr_files:
                # Remove extension
                file_rename = file.replace(".bigwig", "")
                max_dict[file_rename] = max(curr_max) * 1.5

    max_dict = max_dict if custom_max_dict is None else custom_max_dict
    # Read in the ini file
    config = configparser.ConfigParser()
    config.read(ini_path)
    for track in max_dict.keys():
        # Update the max value for each track
        config[track]["max_value"] = str(max_dict[track])

    # Writing the updated ini file
    with open(ini_path, "w") as configfile:
        config.write(configfile)

    print(f"Updated ini file: {ini_path}")
    return ini_path
