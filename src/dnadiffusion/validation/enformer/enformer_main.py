import os
from typing import List
import pickle

import pandas as pd
from tqdm import tqdm

from dnadiffusion.utils.data_util import seq_extract
from dnadiffusion.validation.enformer.enformer import Enformer
from dnadiffusion.validation.enformer.enformerops import EnformerOps

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'


class RunEnformerBase:
    def __init__(
        self,
        region: str,
        tag: str,
        cell_type: str,
        sequence_path: str = 'data/EXPANDED_AXIN2_GM_MASTER_DNA_DIFFUSION_ALL_SEQS.txt',
        model_path: str = 'https://tfhub.dev/deepmind/enformer/1',
        modify_prefix: str = '1_',
    ):
        self.model = Enformer(model_path)
        self.eops = EnformerOps()
        self.tag = tag
        self.modify_prefix = modify_prefix

        # Loading selected sequences into enformer helper class
        sequence_list = seq_extract(sequence_path, tag, cell_type)["SEQUENCE"].values.tolist()
        self.eops.load_data(sequence_list)

        # Loading tracks
        with open("data/tracks_list.pkl", "rb") as f:
            tracks_list = pickle.load(f)
        self.eops.add_track(tracks_list)

        all_sequences = seq_extract(sequence_path, region)
        # Selecting columns of interest
        self.all_sequences = all_sequences[["chrom", "start", "end", "ID"]].values.tolist()
        self.all_sequences = self.all_sequences[:20]

    def extract_dnase(self):
        captured_values = []
        for s in tqdm(self.all_sequences):
            try:
                s_in = [s[0], int(s[1]), int(s[2])]
                id_seq = s[3]
                list_bw = self.eops.generate_plot_number(
                    self.model, 0, interval_list=s_in, wildtype=True, show_track=False, modify_prefix=self.modify_prefix
                )
            except RuntimeError:
                # Infrequent the entries are out of order error for some random seqs
                continue
            try:
                out_in = self.eops.extract_from_position(s_in, as_dataframe=True)
                out_in = out_in.mean()
                out_in["SEQ_ID"] = id_seq
                out_in["TARGET_NAME"] = "ITSELF"
                captured_values.append(out_in)
            except ValueError:
                continue
        df_out = pd.DataFrame([x.values.tolist() for x in captured_values], columns=out_in.index)

        # Save output
        print(f"Saving output to dnase_{self.tag}_seqs.txt")
        df_out.to_csv(f"dnase_{self.tag}_seqs.txt", sep="\t", index=False)


class GeneratedEnformer(RunEnformerBase):
    def __init__(
        self,
        region: str,
        tag: str,
        cell_type: str,
        enhancer_region: str = "chr17_65563685_65564346",
        gene_region: List[str] = ['chr17', "65559702", "65560101"],
        save_interval: int = 10,
    ):
        super().__init__(region, tag, cell_type)
        self.enhancer_region = enhancer_region
        self.gene_region = gene_region
        self.save_interval = save_interval

    def extract_dnase(self):
        captured_values = []
        captured_values_target = []
        for i, s in enumerate(self.all_sequences):
            try:
                s_in = s[1]
                id_seq = s[0]
                self.eops.load_data([id_seq])
                list_bw = self.eops.generate_plot_number(
                    self.model,
                    -1,
                    interval_list=[self.enhancer_region, self.gene_region],
                    wildtype=False,
                    show_track=False,
                    modify_prefix=self.modify_prefix,
                )
            except RuntimeError:
                continue

            try:
                out_in = self.eops.extract_from_position(s_in, as_dataframe=True)
                out_in = out_in.mean()
                out_in["SEQ_ID"] = id_seq
                out_in["TARGET_NAME"] = "ENH_GATA1"
                captured_values.append(out_in)

                out_in = self.eops.extract_from_position(s_in, as_dataframe=True)
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

        print(f"Saving output to {self.modify_prefix}GENERATED_SEQS.TXT")
        df_out.to_csv(self.modify_prefix + "GENERATED_SEQS.TXT", sep="\t", index=False)


if __name__ == "__main__":
    print("Running Enformer")
    RunEnformerBase("RANDOM_GENOME_REGIONS", "GENERATED", "GM12878").extract_dnase()
