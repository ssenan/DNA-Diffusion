import gzip
import shutil
import urllib.request

from dnadiffusion import DATA_DIR

# Download data
print('Downloading hg38')
urllib.request.urlretrieve('http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz', 'data/hg38.fa.gz')
with gzip.open('data/hg38.fa.gz', 'rb') as f_in:
    with open('data/hg38.fa', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

urllib.request.urlretrieve(
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz", f"{DATA_DIR}/clinvar.vcf.gz"
)

urllib.request.urlretrieve(
    "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes", f"{DATA_DIR}/hg38.chrom.sizes"
)

urllib.request.urlretrieve(
    "https://www.dropbox.com/s/a9ggrhn3626x0di/DNA_DIFFUSION_ALL_SEQS.txt?dl=1",
    f"{DATA_DIR}/DNA_DIFFUSION_ALL_SEQS.txt",
)

urllib.request.urlretrieve(
    "https://github.com/pinellolab/DNA-Diffusion/raw/main/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
    f"{DATA_DIR}/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
)

urllib.request.urlretrieve(
    "https://www.dropbox.com/s/oqpn784x34f6pcq/random_regions_train_generated_genome_10k.txt?dl=1",
    f"{DATA_DIR}/random_regions_train_generated_genome_10k.txt",
)

urllib.request.urlretrieve(
    "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/CAGE_peaks/hg38_liftover+new_CAGE_peaks_phase1and2.bed.gz",
    f"{DATA_DIR}/hg38_liftover+new_CAGE_peaks_phase1and2.bed.gz",
)
