import json
import os
from typing import List

import numpy as np
import pandas as pd
import pybedtools
from tqdm import tqdm

from dnadiffusion import DATA_DIR
from dnadiffusion.utils.data_util import seq_extract
from dnadiffusion.validation.enformer.enformer import Enformer
from dnadiffusion.validation.enformer.enformerops import EnformerOps

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

def extract_enhancer_sequence(genome_path:str, chr: str, start: str, end: str, shuffle: bool = False):
    a = pybedtools.BedTool(f"{chr} {start} {end}", from_string=True)
    a = a.sequence(fi=f"{genome_path}")
    sequence =  (open(a.seqfn).read().split("\n")[1])
    if shuffle:
        return ''.join(np.random.permutation(list(sequence)))
    return sequence

class EnformerBase:
    def __init__(
        self,
        region: str | None = None,
        sequence_path: str = f'{DATA_DIR}/validation_dataset.txt',
        model_path: str = 'https://tfhub.dev/deepmind/enformer/1',
        modify_prefix: str = '1_',
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
        with open(f"{DATA_DIR}/tracks_list.json", "r") as f:
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

class LocusVisualization(EnformerBase):
    def __init__(
        self,
        data_path: str = f"{DATA_DIR}/validation_dataset.txt",
        tag: str = "GM12878_positive",
        index: int = 0,
        id: str = "81695_GENERATED_K562",
        genome_path: str = f"{DATA_DIR}/hg38.fa",
        enhancer_region: List[str | int] = ["chrX", 48782929,48783129],
        gene_region: List[str | int] = ['chrX', 48785536, 48787536],
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
                shuffle=self.wildtype_shuffle
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
        cell_type: str | None = None,
        enhancer_region: List[str | int] = ["chrX", 48782929,48783129],
        gene_region: List[str | int] = ['chrX', 48785536, 48787536],
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

        all_sequences = seq_extract(self.sequence_path, tag)
        # generated_seqs = all_sequences[all_sequences["TAG"] == "Generated"]
        self.all_sequences = all_sequences[["SEQUENCE", "ID"]].values.tolist()
        if demo:
            self.all_sequences = self.all_sequences[:50]

    def extract(self):
        file_modify = 1
        captured_values = []
        captured_values_target = []
        for i, s in tqdm(enumerate(self.all_sequences), total=len(self.all_sequences)):
            try:
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
            except RuntimeError:
                continue

            try:
                # out_in = self.eops.extract_from_position(self.enhancer_region, as_dataframe=True)
                # out_in = out_in.mean()
                # out_in["SEQ_ID"] = s_in
                # out_in["TARGET_NAME"] = "ENH_GATA1"
                # captured_values.append(out_in)
                #
                # out_in = self.eops.extract_from_position(self.gene_region, as_dataframe=True)
                # out_in = out_in.mean()
                # out_in["SEQ_ID"] = id_seq
                # out_in["TARGET_NAME"] = "GATA1_TSS_2K"
                # captured_values_target.append(out_in)
                pass
            except ValueError:
                continue

            # if (i != 0) and ((i+1) % self.save_interval) == 0:
            #     df_out_ENH = pd.DataFrame(
            #         [x.values.tolist() for x in captured_values], columns=["ENHANCER_" + x for x in out_in.index]
            #     )
            #     df_out_GENE = pd.DataFrame(
            #         [x.values.tolist() for x in captured_values_target], columns=["GENE_" + x for x in out_in.index]
            #     )
            #
            #     df_out = pd.concat([df_out_ENH, df_out_GENE], axis=1)
            #
            #     df_out.to_csv(f"{DATA_DIR}/{str(file_modify)}_test_seqs.TXT", sep="\t", index=False)
            #     # Resetting captured values
            #     captured_values = []
            #     captured_values_target = []
            #     file_modify += 1

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


if __name__ == "__main__":
    #EnformerBase(region="Promoters").extract()
    # EnformerBase(region="Random_Genome_Regions", demo=True).extract()
    # for tag in ["Generated", "GM12878_positive", "HepG2_positive", "Test", "Training", "Validation", "Negative"]:
    # print(f"Running Enformer for {tag}")
    # print(10 *"=")
    # GeneratedEnformer(tag=tag).extract()
    GeneratedEnformer(tag="Generated", demo=True).extract()
