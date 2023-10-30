import json
import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from dnadiffusion import DATA_DIR
from dnadiffusion.utils.data_util import seq_extract
from dnadiffusion.validation.enformer.enformer import Enformer
from dnadiffusion.validation.enformer.enformer_utils import normalize_tracks
from dnadiffusion.validation.enformer.enformerops import EnformerOps
from dnadiffusion.validation.validation_utils import extract_enhancer_sequence

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"


class EnformerBase:
    def __init__(
        self,
        region: str | None = None,
        sequence_path: str = f"{DATA_DIR}/validation_dataset.txt",
        model_path: str = "https://tfhub.dev/deepmind/enformer/1",
        modify_prefix: str = "1_",
        sequence_length: int = 393216,
        show_track: bool = False,
        demo: bool = False,
    ):
        self.model = Enformer(model_path)
        self.eops = EnformerOps()
        self.sequence_path = sequence_path
        self.region = region
        self.modify_prefix = modify_prefix
        self.show_track = show_track

        chrom_sizes = pd.read_table(f"{DATA_DIR}/hg38.chrom.sizes", header=None).set_index(0).to_dict()[1]

        # Loading tracks from json
        with open(f"{DATA_DIR}/tracks_list.json") as f:
            tracks_list = json.load(f)
        self.eops.add_track(tracks_list)

        all_sequences = seq_extract(sequence_path, region)
        # Selecting columns of interest
        self.all_sequences = all_sequences[["chrom", "start", "end", "ID"]].values.tolist()
        self.all_sequences = [
            x
            for x in self.all_sequences
            if (int(x[1]) > sequence_length / 2) and (np.abs(int(chrom_sizes[x[0]]) - int(x[2])) > sequence_length / 2)
        ]
        if demo:
            print(len(self.all_sequences))
            self.all_sequences = self.all_sequences[2050:2070]

    def extract(self):
        captured_values = []
        for s in tqdm(self.all_sequences):
            s_in = [s[0], int(s[1]), int(s[2])]
            id_seq = s[3]
            self.eops.generate_tracks(
                self.model,
                0,
                interval_list=s_in,
                wildtype=True,
                show_track=self.show_track,
                modify_prefix=self.modify_prefix,
            )
            out_in = self.eops.extract_from_position(s_in, as_dataframe=True)
            out_in = out_in.mean()
            out_in["SEQ_ID"] = id_seq
            out_in["TARGET_NAME"] = "ITSELF"
            captured_values.append(out_in)
        df_out = pd.DataFrame([x.values.tolist() for x in captured_values], columns=out_in.index)

        # Save output
        print(f"Saving output to dnase_{self.region}_seqs.txt")
        df_out.to_csv(f"{DATA_DIR}/{self.region}_seqs.txt", sep="\t", index=False)

    def capture_normalization_values(
        self,
        tag: str,
        track_name: str,
        amount: int,
    ):
        """Capture values to be used for during quantile normalization of enformer tracks
        Args:
            sequence_path (str): Sequence path
            tag (str): Tag to extract sequences (e.g. Random_genome_regions or Promoters)
            amount (int): Amount of sequences to extract

        Returns:
            pd.DataFrame: DataFrame with values to be used for normalization
        """
        # Creating sample set of input tag dataframe
        sequence_df = seq_extract(self.sequence_path, tag).sample(amount)

        # Loading sequences into enformerops
        sequences = sequence_df["SEQUENCE"].values.tolist()

        # Selecting columns of interest
        sequence_df = sequence_df[["chrom", "start", "end"]].values.tolist()

        final_output = []

        for chr_x, start_x, end_x in tqdm(sequence_df):
            for seq in sequences:
                self.eops.load_data([seq])
            # Defining region to extract from
            sequence_region = [chr_x, int(start_x), int(end_x)]
            self.eops.generate_tracks(
                self.model,
                -1,
                interval_list=sequence_region,
                wildtype=True,
                show_track=self.show_track,
                modify_prefix=self.modify_prefix,
            )

            # Modifying sequence region to extract from
            mod_start = int(sequence_region[1] + ((sequence_region[2] - sequence_region[1]) / 2)) - int(114688 / 2)
            mod_end = int(sequence_region[1] + ((sequence_region[2] - sequence_region[1]) / 2)) + int(114688 / 2)
            mod_sequence_region = [sequence_region[0], mod_start, mod_end]

            # Extracting values from tracks
            extracted_values = self.eops.extract_from_position(mod_sequence_region, as_dataframe=True)
            final_output.append(extracted_values)

        # Clearing sequence data from enformerops
        self.eops.clear_data()

        df_out = pd.concat(final_output)
        # Only selecting columns that contain "DNASE" in their name
        df_out = df_out[df_out.columns[df_out.columns.str.contains(f"{track_name}")]]
        df_out.to_csv(f"{DATA_DIR}/{track_name}_normalization_values.txt", sep="\t", index=False)


class LocusVisualization(EnformerBase):
    def __init__(
        self,
        data_path: str = f"{DATA_DIR}/validation_dataset.txt",
        tag: str = "GM12878_positive",
        index: int = 0,
        id: str = "81695_GENERATED_K562",
        genome_path: str = f"{DATA_DIR}/hg38.fa",
        enhancer_region: List[str | int] = ["chrX", 48782929, 48783129],
        gene_region: List[str | int] = ["chrX", 48785536, 48787536],
        show_track: bool = False,
        wildtype_shuffle: bool = False,
        input_sequence: str | None = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.tag = tag
        self.index = index
        self.id = id
        self.genome_path = genome_path
        self.enhancer_region = enhancer_region
        self.gene_region = gene_region
        self.show_track = show_track
        self.wildtype_shuffle = wildtype_shuffle
        self.input_sequence = input_sequence

    def extract(self):
        captured_values = []
        captured_values_target = []
        if self.input_sequence:
            sequences = seq_extract(self.data_path, f"{self.tag}")
            id_seq = sequences[sequences.ID == f"{self.id}"]["SEQUENCE"].values[0]

        else:
            id_seq = extract_enhancer_sequence(
                self.genome_path,
                self.enhancer_region[0],
                str(self.enhancer_region[1]),
                str(self.enhancer_region[2]),
                shuffle=self.wildtype_shuffle,
            )

        self.eops.load_data([id_seq])
        self.eops.generate_tracks(
            self.model,
            -1,
            interval_list=self.enhancer_region,
            wildtype=False,
            gene_region=self.gene_region,
            show_track=self.show_track,
            modify_prefix=self.modify_prefix,
        )


class GeneratedEnformer(EnformerBase):
    def __init__(
        self,
        tag: str,
        enhancer_region: List[str | int] = ["chrX", 48782929, 48783129],
        gene_region: List[str | int] = ["chrX", 48785536, 48787536],
        save_interval: int = 10,
        show_track: bool = False,
        demo: bool = False,
    ):
        super().__init__()
        self.tag = tag
        self.enhancer_region = enhancer_region
        self.gene_region = gene_region
        self.save_interval = save_interval
        self.show_track = show_track

        # Values for normalization
        self.normalized_dnase_path = normalized_dnase_path
        self.normalized_cage_path = normalized_cage_path

        all_sequences = seq_extract(self.sequence_path, tag)
        self.all_sequences = all_sequences[["SEQUENCE", "ID"]].values.tolist()

        if demo:
            self.all_sequences = self.all_sequences[:50]

    def extract(self):
        file_modify = 1
        captured_values = []
        captured_values_target = []
        for i, s in tqdm(enumerate(self.all_sequences), total=len(self.all_sequences)):
            s_in = s[1]
            id_seq = s[0]
            self.eops.load_data([id_seq])
            self.eops.generate_tracks(
                self.model,
                -1,
                interval_list=self.enhancer_region,
                wildtype=False,
                gene_region=self.gene_region,
                show_track=self.show_track,
                modify_prefix=self.modify_prefix,
            )

            out_in = self.eops.extract_from_position(self.enhancer_region, as_dataframe=True)
            out_in = out_in.mean()
            out_in["SEQ_ID"] = s_in
            out_in["TARGET_NAME"] = "ENH_GATA1"
            captured_values.append(out_in)

            out_in = self.eops.extract_from_position(self.gene_region, as_dataframe=True)
            out_in = out_in.mean()
            out_in["SEQ_ID"] = id_seq
            out_in["TARGET_NAME"] = "GATA1_TSS_2K"
            captured_values_target.append(out_in)

            if (i != 0) and ((i + 1) % self.save_interval) == 0:
                df_out_ENH = pd.DataFrame(
                    [x.values.tolist() for x in captured_values], columns=["ENHANCER_" + x for x in out_in.index]
                )
                df_out_GENE = pd.DataFrame(
                    [x.values.tolist() for x in captured_values_target], columns=["GENE_" + x for x in out_in.index]
                )

                df_out = pd.concat([df_out_ENH, df_out_GENE], axis=1)

                df_out.to_csv(f"{DATA_DIR}/{file_modify}_test_seqs.TXT", sep="\t", index=False)
                # Resetting captured values
                captured_values = []
                captured_values_target = []
                file_modify += 1

        # Find all the files written to DATA_DIR from the above loop
        files = sorted([f"{DATA_DIR}/{f}" for f in os.listdir(DATA_DIR) if f.endswith("_test_seqs.TXT")])

        df_out = pd.concat([pd.read_csv(f, sep="\t") for f in files], axis=0)

        # Save output
        print(f"Saving output to {DATA_DIR}/{self.tag}_test_seqs_final.TXT")
        df_out.to_csv(f"{DATA_DIR}/{self.tag}_test_seqs_final.TXT", sep="\t", index=False)

        # Remove all the files written to DATA_DIR from the above loop
        for f in files:
            os.remove(f)

        # df_out_ENH = pd.DataFrame(
        #     [x.values.tolist() for x in captured_values], columns=["ENHANCER_" + x for x in out_in.index]
        # )
        # df_out_GENE = pd.DataFrame(
        #     [x.values.tolist() for x in captured_values_target], columns=["GENE_" + x for x in out_in.index]
        # )
        #
        # df_out = pd.concat([df_out_ENH, df_out_GENE], axis=1)
        #
        # print(f"Saving output to {DATA_DIR}/{self.tag}_GENERATED_SEQS.TXT")
        # df_out.to_csv(f"{DATA_DIR}/" + self.tag + "_GENERATED_SEQS.TXT", sep="\t", index=False)


class NormalizeTracks(EnformerBase):
    def __init__(
        self,
        input_sequence: str,
        enhancer_region: List[str | int] = ["chrX", 48782929, 48783129],
        gene_region: List[str | int] = ["chrX", 48785536, 48787536],
        normalized_dnase_path: str = f"{DATA_DIR}/DNASE_normalization_values.txt",
        normalized_cage_path: str = f"{DATA_DIR}/CAGE_normalization_values.txt",
        normalized_h3k4me3_path: str = f"{DATA_DIR}/H3K4ME3_normalization_values.txt",
    ):
        super().__init__()
        self.input_sequence = input_sequence
        self.enhancer_region = enhancer_region
        self.gene_region = gene_region
        self.normalized_dnase_rgr = pd.read_csv(normalized_dnase_path, sep="\t")
        self.normalized_cage_promoters = pd.read_csv(normalized_cage_path, sep="\t")
        self.normalized_h3k4me3_promoters = pd.read_csv(normalized_h3k4me3_path, sep="\t")

    def extract(self):
        # Processing sequence of interest
        self.eops.load_data([self.input_sequence])
        # Creating tracks
        self.eops.generate_tracks(self.model, -1, interval_list=self.enhancer_region, wildtype=False, show_track=False)
        # Extracting values from tracks
        mod_start = int(self.enhancer_region[1] + ((self.enhancer_region[2] - self.enhancer_region[1]) / 2)) - int(
            114688 / 2
        )
        mod_end = int(self.enhancer_region[1] + ((self.enhancer_region[2] - self.enhancer_region[1]) / 2)) + int(
            114688 / 2
        )
        mod_sequence_region = [self.enhancer_region[0], mod_start, mod_end]
        extracted_values = self.eops.extract_from_position(mod_sequence_region, as_dataframe=True)

        # Normalizing DNASE values
        extracted_values_dnase = extracted_values[[x for x in extracted_values.columns if "DNASE" in x]]
        normalized_dnase = normalize_tracks(extracted_values_dnase, self.normalized_dnase_rgr)
        # Normalizing CAGE values
        extracted_values_cage = extracted_values[[x for x in extracted_values.columns if "CAGE" in x]]
        normalized_cage = normalize_tracks(extracted_values_cage, self.normalized_cage_promoters)
        # Normalizing H3K4ME3 values
        extracted_values_h3k4me3 = extracted_values[[x for x in extracted_values.columns if "H3K4ME3" in x]]
        normalized_h3k4me3 = normalize_tracks(extracted_values_h3k4me3, self.normalized_h3k4me3_promoters)

        # Concatenating normalized values
        normalized_values = pd.concat([normalized_dnase, normalized_cage, normalized_h3k4me3], axis=1)

        self.eops.generate_normalized_tracks(
            normalized_values,
            self.gene_region,
            self.enhancer_region,
        )
        print("Finished normalizing tracks")


if __name__ == "__main__":
    # EnformerBase(region="Promoters").extract()
    # EnformerBase(region="Random_Genome_Regions", demo=True).extract()
    # for tag in ["Generated", "GM12878_positive", "HepG2_positive", "Test", "Training", "Validation", "Negative"]:
    # print(f"Running Enformer for {tag}")
    # print(10 *"=")
    # GeneratedEnformer(tag=tag).extract()
    # GeneratedEnformer(tag="Generated", demo=True).extract()
    # EnformerBase(region="Random_Genome_Regions").capture_normalization_values(
    #     tag="Promoters", track_name="CAGE", amount=100
    # )
    NormalizeTracks(
        input_sequence="GCAACTTACAACCACAGAATTCAGTTCTCAAAATAGGACACAGAGAAAGTGAGACTGAGAAGTGTGGAAATTCCCCCAGCCTGTCGGACTGGACTAATGTTTCATTCGTAATTAGGTACAAAAAAGCCATCAGTACAGTGGAAAGCAGGGAGTTCAGATGTGACATATAATTCTTTTTCCCTATTCACTTTCTCTTCCCT"
    ).extract()
