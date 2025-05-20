import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dnadiffusion.utils.utils import convert_to_seq


def create_sample(
    model: torch.nn.Module,
    cell_types: list[int],
    sample_bs: int,
    conditional_numeric_to_tag: dict,
    number_of_samples: int = 1000,
    group_number: list | None = None,
    cond_weight_to_metric: int = 0,
    save_timesteps: bool = False,
    save_dataframe: bool = False,
    generate_attention_maps: bool = False,
    continuous_vector: list[float] = None,
) -> None:
    nucleotides = ["A", "C", "G", "T"]
    final_sequences = []
    num_batches = number_of_samples // sample_bs
    
    # Check if we're using continuous conditioning
    use_continuous = hasattr(model.model, 'use_continuous_conditioning') and model.model.use_continuous_conditioning
    
    for n_a in tqdm(range(num_batches)):
        if use_continuous and continuous_vector is not None:
            # If a continuous vector is provided, use it
            print(f"Using continuous vector: {continuous_vector}")
            continuous_tensor = torch.tensor([continuous_vector] * sample_bs, dtype=torch.float)
            classes = continuous_tensor.to(model.device)
        elif group_number:
            if use_continuous:
                # If continuous conditioning but with a specific cell type,
                # create a one-hot vector for that cell type
                num_classes = len(conditional_numeric_to_tag)
                one_hot = torch.zeros(sample_bs, num_classes)
                # Convert cell index to one-hot position
                idx = cell_types.index(group_number) if group_number in cell_types else 0
                one_hot[:, idx] = 1.0
                classes = one_hot.to(model.device)
                print(f"Using one-hot vector for cell type {conditional_numeric_to_tag[group_number]}: {one_hot[0].tolist()}")
            else:
                # Traditional discrete conditioning
                sampled = torch.from_numpy(np.array([group_number] * sample_bs))
                classes = sampled.float().to(model.device)
                print(f"Using discrete conditioning with cell type: {conditional_numeric_to_tag[group_number]}")
        else:
            if use_continuous:
                # Random cell type with continuous conditioning
                num_classes = len(conditional_numeric_to_tag)
                sampled_indices = np.random.choice(len(cell_types), sample_bs)
                one_hot = torch.zeros(sample_bs, num_classes)
                for i, idx in enumerate(sampled_indices):
                    one_hot[i, idx] = 1.0
                classes = one_hot.to(model.device)
                print("Using random continuous conditioning (one-hot vectors)")
            else:
                # Random cell type with discrete conditioning
                sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))
                classes = sampled.float().to(model.device)
                print("Using random discrete conditioning")

        if generate_attention_maps:
            sampled_images, cross_att_values = model.sample_cross(
                classes, (sample_bs, 1, 4, 200), cond_weight_to_metric
            )
            # save cross attention maps in a numpy array
            np.save(f"cross_att_values_{conditional_numeric_to_tag[group_number]}.npy", cross_att_values)

        else:
            sampled_images = model.sample(classes, (sample_bs, 1, 4, 200), cond_weight_to_metric)

        if save_timesteps:
            seqs_to_df = {}
            for en, step in enumerate(sampled_images):
                seqs_to_df[en] = [convert_to_seq(x, nucleotides) for x in step]
            final_sequences.append(pd.DataFrame(seqs_to_df))

        if save_dataframe:
            # Only using the last timestep
            for en, step in enumerate(sampled_images[-1]):
                final_sequences.append(convert_to_seq(step, nucleotides))
        else:
            for n_b, x in enumerate(sampled_images[-1]):
                seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
                    [nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)]
                )
                final_sequences.append(seq_final)

    # Determine the filename based on conditioning type
    if continuous_vector is not None:
        # For continuous vector, create a filename based on the vector values
        filename = "continuous_" + "_".join([f"{v:.2f}" for v in continuous_vector])
    elif group_number:
        # For discrete group, use the group name
        filename = conditional_numeric_to_tag[group_number]
    else:
        # For random sampling
        filename = "random_sampling"
        
    if save_timesteps:
        # Saving dataframe containing sequences for each timestep
        pd.concat(final_sequences, ignore_index=True).to_csv(
            f"data/outputs/{filename}.txt",
            header=True,
            sep="\t",
            index=False,
        )
        return

    if save_dataframe:
        # Saving list of sequences to txt file
        with open(f"data/outputs/{filename}.txt", "w") as f:
            f.write("\n".join(final_sequences))
        return
