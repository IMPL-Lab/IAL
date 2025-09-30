import os
import numpy as np

def get_lidar_token(nusc, sample_token):
    """ Get lidar token from given sample token using nuscenes toolkit. """
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    return lidar_token

def save_test_results(
    sem_preds, ins_preds, output_dir, batch, class_inv_lut, tokens, dataset
):
    if dataset == "semantickitti":
        for i in range(len(sem_preds)):
            sem = sem_preds[i]
            ins = ins_preds[i]
            sem_inv = class_inv_lut[sem].astype(np.uint32)
            label = sem_inv.reshape(-1, 1) + (
                (ins.astype(np.uint32) << 16) & 0xFFFF0000
            ).reshape(-1, 1)

            pcd_path = batch["fname"][i]
            seq = pcd_path.split("/")[-3]
            pcd_fname = pcd_path.split("/")[-1].split(".")[-2] + ".label"
            fname = os.path.join(output_dir, seq, "predictions", pcd_fname)
            label.reshape(-1).astype(np.uint32).tofile(fname)
    elif dataset == "nuscenes":
        sem = sem_preds
        ins = ins_preds
        label = 1000 * sem.reshape(-1, 1) + ins.astype(np.uint32).reshape(-1, 1)
        name = tokens + "_panoptic.npz"
        fname = os.path.join(output_dir, name)
        np.savez_compressed(fname, data=label.reshape(-1).astype(np.uint16))
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset}")
    
    
