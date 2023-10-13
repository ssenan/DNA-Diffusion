import os
import pickle
from typing import List

import pandas as pd
from tqdm import tqdm

from dnadiffusion import DATA_DIR
from dnadiffusion.utils.data_util import seq_extract
from dnadiffusion.validation.enformer.enformer import Enformer
from dnadiffusion.validation.enformer.enformerops import EnformerOps

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'


class EnformerBase:
    def __init__(
        self,
        region: str | None = None,
        sequence_path: str = f'{DATA_DIR}/validation_dataset.txt',
        model_path: str = 'https://tfhub.dev/deepmind/enformer/1',
        modify_prefix: str = '1_',
        show_track: bool = False,
        demo: bool = False,
    ):
        self.model = Enformer(model_path)
        self.eops = EnformerOps()
        self.sequence_path = sequence_path
        self.region = region
        self.modify_prefix = modify_prefix
        self.show_track = show_track

        # Loading selected sequences into enformer helper class
        #sequence_list = seq_extract(sequence_path, tag, cell_type)["SEQUENCE"].values.tolist()
        #self.eops.load_data(sequence_list)

        # Loading tracks
        with open(f"{DATA_DIR}/tracks_list.pkl", "rb") as f:
            tracks_list = pickle.load(f)
        self.eops.add_track(tracks_list)

        all_sequences = seq_extract(sequence_path, region)
        # Selecting columns of interest
        self.all_sequences = all_sequences[["chrom", "start", "end", "ID"]].values.tolist()
        if demo:
            self.all_sequences = [self.all_sequences[0]]


    def extract(self):
        captured_values = []
        for s in tqdm(self.all_sequences):
            try:
                s_in = [s[0], int(s[1]), int(s[2])]
                id_seq = s[3]
                self.eops.generate_plot_number(
                    self.model, 0, interval_list=s_in, wildtype=True, show_track=self.show_track, modify_prefix=self.modify_prefix
                )
            except RuntimeError as r:
                # Infrequent the entries are out of order error for some random seqs
                continue

            try:
                out_in = self.eops.extract_from_position(s_in, as_dataframe=True)
                out_in = out_in.mean()
                out_in["SEQ_ID"] = id_seq
                out_in["TARGET_NAME"] = "ITSELF"
                captured_values.append(out_in)
            except ValueError as v:
                continue
        df_out = pd.DataFrame([x.values.tolist() for x in captured_values], columns=out_in.index)

        # Save output
        print(f"Saving output to dnase_{self.region}_seqs.txt")
        df_out.to_csv(f"{DATA_DIR}/{self.region}_seqs.txt", sep="\t", index=False)


class GeneratedEnformer(EnformerBase):
    def __init__(
        self,
        tag: str,
        cell_type: str | None = None,
        enhancer_region: List[str | int] = ["chr17", 65563685, 65564346],
        gene_region: List[str | int] = ["chr17", 65559702, 65560101],
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
            self.all_sequences = self.all_sequences[:10]

    def extract(self):
        captured_values = []
        captured_values_target = []
        for i, s in tqdm(enumerate(self.all_sequences)):
            try:
                s_in = s[1]
                id_seq = s[0]
                self.eops.load_data([id_seq])
                self.eops.generate_plot_number(
                    self.model,
                    -1,
                    interval_list=self.enhancer_region,
                    wildtype=False,
                    show_track=self.show_track,
                    modify_prefix=self.modify_prefix,
                )
            except RuntimeError:
                continue

            try:
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
            except ValueError:
                continue

            if (i != 0) and (i % self.save_interval) == 0:
                df_out_ENH = pd.DataFrame(
                    [x.values.tolist() for x in captured_values], columns=["ENHANCER_" + x for x in out_in.index]
                )
                df_out_GENE = pd.DataFrame(
                    [x.values.tolist() for x in captured_values_target], columns=["GENE_" + x for x in out_in.index]
                )

                df_out = pd.concat([df_out_ENH, df_out_GENE], axis=1)

                df_out.to_csv(self.modify_prefix + "GENERATED_SEQS.TXT", sep="\t", index=False)

        df_out_ENH = pd.DataFrame(
            [x.values.tolist() for x in captured_values], columns=["ENHANCER_" + x for x in out_in.index]
        )
        df_out_GENE = pd.DataFrame(
            [x.values.tolist() for x in captured_values_target], columns=["GENE_" + x for x in out_in.index]
        )

        df_out = pd.concat([df_out_ENH, df_out_GENE], axis=1)

        print(f"Saving output to {DATA_DIR}/{self.tag}_GENERATED_SEQS.TXT")
        df_out.to_csv(f"{DATA_DIR}/" + self.tag + "_GENERATED_SEQS.TXT", sep="\t", index=False)


if __name__ == "__main__":
    #EnformerBase(region="Promoters").extract()
    #EnformerBase(region="Random_Genome_Regions").extract()
    for tag in ["Generated", "GM12878_positive", "HepG2_positive", "Test", "Training", "Validation", "Negative"]:
        print(f"Running Enformer for {tag}")
        print(10 *"=")
        GeneratedEnformer(tag=tag).extract()
    GeneratedEnformer(tag="Test", demo=True).extract()
