from dataclasses import dataclass, field
from typing import Dict, List

import igv_notebook
import kipoiseq
import numpy as np
import pandas as pd
import pyBigWig

from dnadiffusion.validation.enformer.enformer import Enformer, FastaStringExtractor
from dnadiffusion.validation.enformer.enformer_utils import one_hot_encode
from dnadiffusion import DATA_DIR

@dataclass
class EnformerData:
    tracks: List = field(default_factory=list)
    input_sequences_file_path: str = field(default_factory=str)
    interval_list: List = field(default_factory=list)
    capture_bigwig_names: bool = field(default_factory=bool)
    full_generated_range_start: int = field(default_factory=int)
    full_generated_range_end: int = field(default_factory=int)
    loaded_seqs: List = field(default_factory=list)


class EnformerOps(EnformerData):
    def __init__(self):
        super().__init__()
        self.df_sizes: pd.DataFrame = pd.read_table('data/hg38.chrom.sizes', header=None).head(22)
        self.fasta_extractor: FastaStringExtractor = FastaStringExtractor('data/hg38.fa')

    def add_track(self, additional_tracks: Dict | List[Dict]):
        """
        Adds a track to the list of tracks to be visualized.

        Args:
            track (dict): A dictionary specifying the track to be added.
            Should have the keys "name", "file", "color", "type", and "id"
            (if type is "enformer").
        """
        if type(additional_tracks) == dict:
            self.tracks.append(additional_tracks)
        elif type(additional_tracks) == list:
            self.tracks.extend(additional_tracks)
        else:
            raise TypeError("add_track must be of type dict or list")

    def remove_track(self, track_key):
        """
        Removes track from track list.

        Args:
            track_key (str or list): Name of the track(s) to be deleted
        """

        if type(track_key) == str:
            self.tracks.remove(track_key)
        elif type(track_key) == list:
            for key in track_key:
                self.tracks.remove(key)
        else:
            raise TypeError("track_key must be of type str or list")

    def load_data(self, input_sequences: List[str]):
        if type(input_sequences) == list:
            self.loaded_seqs = [[x] for x in input_sequences]
            self.input_sequences_file_path = input_sequences
        else:
            TypeError("input_sequence_path must be of type list")

    def generate_plot_number(
        self,
        model: Enformer,
        sequence_number_thousand: int,
        step: int = -1,
        interval_list: str | List[str] | None = None,
        show_track: bool = False,
        wildtype: bool = False,
        modify_prefix: str = "",
        sequence_length: int = 393216,
        replace_remove_region: bool = False,
    ):
        """
        Generates IGV tracks for a given sequence in a diffusion dataset.

        Args:
            sequence_number_thousand (int): The number of the sequence ID in
            the diffusion sequences FASTA dataset.

            step (int, optional): Which diffusion step to use. Default is -1,
            which means the last diffusion step (i.e., the final diffused sequence).

            interval_list (list, optional): Coordinate to insert the 200 bp
            sequence. Should be in BED format (chr, start, end). Default is None. This is only used during GeneratedEnformer class call.

            show_track (bool, optional): Whether to generate IGV tracks as a result.
            Default is True.

            capture_bigwig_names (bool, optional): Whether to output a list with
            all IGV tracks generated and used (in case real bigwig files were used)
            for the final visualization. Default is True.

            wildtype (bool, False)
            Don't insert and capture the wildtype sequence

        Returns:
            list: A list with the name of all bigwig files generated.
        """
        # List to capture all bigwig files generated
        bigwig_names = []
        if not interval_list:
            interval_list = self.interval_list

        # Get the sequences and target interval
        target_interval = kipoiseq.Interval(*interval_list)
        # difference = 0

        """if replace_remove_region:
            seq_overwrite = self.fasta_extractor.extract(target_interval)
            len_seq_add = len(seqs_test[step])
            difference = int(round((len(seq_overwrite) - len_seq_add) / 2))
            add_nucleotide = 1
            if len_seq_add % 4 != 0:
                add_nucleotide = -1
            difference = (difference * 2) + add_nucleotide
        """
        chr_test = target_interval.resize(sequence_length).chr
        start_test = target_interval.resize(sequence_length).start
        end_test = target_interval.resize(sequence_length).end

        """if replace_remove_region:
            seq_to_modify = self.fasta_extractor.extract(target_interval.resize(sequence_length))
            seq_input = seq_to_modify.replace(seq_overwrite, seqs_test[step])
        """
        seq_to_modify = self.fasta_extractor.extract(target_interval.resize(sequence_length)) 
        if wildtype:
            seq_input = self.insert_seq(seq_to_modify, dont_insert=wildtype)
        else:
            seqs_test = self.loaded_seqs[sequence_number_thousand]
            seq_input = self.insert_seq(seq_to_modify, seqs_test[step], dont_insert=wildtype)

        predictions = self.predict_from_sequence(model, seq_input)
        mod_start = int(start_test + ((end_test - start_test) / 2) - int(114688 / 2))
        mod_end = int(start_test + ((end_test - start_test) / 2) + int(114688 / 2))

        self.full_generated_range_start = mod_start
        self.full_generated_range_end = mod_end
        self.full_generated_chr = chr_test

        if show_track:
            igv_notebook.init()
            b = igv_notebook.Browser(
                {
                    "genome": "hg38",
                    "locus": f"{chr_test}:{mod_start}-{mod_end}",
                }
            )

        for track in self.tracks:
            if track["type"] == "enformer":
                id = track["id"]
                n = modify_prefix + track["name"]
                lg = track["log"]

                p_values = predictions[:, id]
                if lg == True:
                    p_values = np.log10(1 + predictions[:, id])
                bigwig_names.append(n + ".bw")
                out_track = self._enformer_bigwig_creation(chr_test, mod_start, p_values, n)
                if show_track:
                    b.load_track(out_track)

            elif track["type"] == "real":
                n = track["name"]
                f = modify_prefix + track["file"]
                c = track["color"]
                bigwig_names.append(f)
                if show_track:
                    b.load_track(self._generate_real_tracks(n, f, c))

        self.bigwig_names = bigwig_names
        return bigwig_names

    def extract_from_position(self, position, as_dataframe=False):
        """
        Extracts data from the bigwig files generated by generate_plot_number for a given genomic region.

        Args:
            chr_name (str): The name of the chromosome.
            start (int): The start position of the region.
            end (int): The end position of the region.

        Returns:
            list: A list of dictionaries containing the name of each bigwig file
            and the values for the given region.
        """
        if self.capture_bigwig_names is None:
            raise ValueError("Must call generate_plot_number first to generate the bigwig files.")

        results = []

        for name in self.bigwig_names:
            bw = pyBigWig.open(name)
            values = bw.values(position[0], position[1], position[2])
            results.append({'name': name, 'values': values})
        if as_dataframe:
            results = pd.DataFrame({k['name']: k['values'] for k in results})

        return results

    @staticmethod
    def predict_from_sequence(model, input_sequence):
        sequence_one_hot = one_hot_encode(input_sequence)
        return model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]

    @staticmethod
    def insert_seq(seq_mod_in: str, seq_x: str | None = None, dont_insert: bool =False):
        '''
        This function inserts a sequence `seq_x` into a larger sequence `seq_mod_in`.

        Args:
            seq_x (str): The sequence to be inserted into `seq_mod_in`.
            seq_mod_in (str): The larger sequence that `seq_x` will be inserted into.
            dont_insert (bool, optional): Whether or not to skip inserting `seq_x`.
            If `True`, `seq_mod_in` will be returned unchanged. Default is `False`.

        Returns:
            str: The modified sequence with `seq_x` inserted into `seq_mod_in`.
        '''
        seq_to_mod_array = np.array(list(seq_mod_in))
        seq_mod_center = seq_to_mod_array.shape[0] // 2
        if not dont_insert:
            seq_to_mod_array[seq_mod_center - 100 : seq_mod_center + 100] = np.array(list(seq_x))

        return ''.join(seq_to_mod_array)

    def _enformer_bigwig_creation(self, chr_name, start, values, track_name, color='BLUE'):
        """
        Creates a bigwig file for an Enformer track.

        Args:
            chr_name (str): The name of the chromosome.
            start (int): The start position of the track.
            values (np.array): The values to be used in the track.
            track_name (str): The name of the track.
            color (str, optional): The color to use for the track. Default is 'BLUE'.

        Returns:
            dict: A dictionary containing the name and path of the bigwig file,
            as well as its format, display mode, and color.
        """

        t_name = f"{track_name}.bw"
        bw = pyBigWig.open(t_name, "w")
        bw.addHeader([(chr_name, coord) for chr_name, coord in self.df_sizes.values])
        values_conversion = (values * 1000).astype(np.int64) + 0.0
        bw.addEntries(
            chr_name, [start + (128 * x) for x in range(values_conversion.shape[0])], values=values_conversion, span=128
        )

        return {
            "name": f"{track_name}",
            "path": f"{track_name}.bw",
            "format": "bigwig",
            "displayMode": "EXPANDED",
            "color": f"{color}",
            "height": 100,
        }

    def _generate_real_tracks(self, name, filename, color):
        """
        Generates a real track for a given bigwig file.

        Args:
            name (str): The name of the track.
            file (str): The name of the bigwig file to use for the track.
            color (str): The color to use for the track.


        Returns:
            dict: A dictionary containing the name, path, format, display mode, and color of the track.
        """

        chr_name, start, end = self.full_generated_chr, self.full_generated_range_start, self.full_generated_range_end
        t_name = f"{name}_minimal.bw"
        bw = pyBigWig.open(t_name, "w")
        bw.addHeader([(chr_name, coord) for chr_name, coord in self.df_sizes.values])
        bw_cut = pyBigWig.open(filename, "r")
        values = np.array(bw_cut.values(chr_name, start, end))

        values_conversion = (values * 1000).astype(np.int64) + 0.0
        print(values_conversion)
        print(chr_name, start)
        bw.addEntries(
            chr_name,
            [r for r in range(start, start + len(values_conversion))],
            values=list(values_conversion),
            span=1,
            step=1,
        )
        bw.close()
        bw_cut.close()

        return {
            "name": name,
            "path": t_name,
            "format": "bigwig",
            "displayMode": "EXPANDED",
            "color": color,
            "height": 100,
        }
