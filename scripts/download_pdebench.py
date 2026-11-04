"""Download the PDEBench compressible Navier–Stokes files used in the paper.

    python scripts/download_pdebench.py --root data/pdebench              # CNS-2D and CNS-3D
    python scripts/download_pdebench.py --root data/pdebench --dataset cns_3d

Files land where the configs expect them:
    <root>/2D/CFD/2D_Train_Turb/   CNS-2D, 512 x 512, Mach 0.1 and 1.0
    <root>/3D/Train/               CNS-3D, Mach 1.0
"""

import argparse
import hashlib
import os
import urllib.request

DARUS = "https://darus.uni-stuttgart.de/api/access/datafile/"

# dataset -> [(directory, file name, DaRUS file id, md5)]
FILES = {
    "cns_2d": [
        ("2D/CFD/2D_Train_Turb", "2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5", 164685, "844555000d342d2947162c6cf46798e7"),
        ("2D/CFD/2D_Train_Turb", "2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5", 164686, "3f2c7376cde5fb072db0f9814f1c6992"),
    ],
    "cns_3d": [
        ("3D/Train", "3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", 164694, "4f0eacaec5ff000bbd9a31026ea207b8"),
    ],
}


def md5sum(path: str, chunk_size: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: str, directory: str, filename: str, file_id: int, md5: str) -> None:
    path = os.path.join(root, directory, filename)
    if os.path.exists(path) and md5sum(path) == md5:
        print(f"{path} already downloaded")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(DARUS + str(file_id), path)
    if md5sum(path) != md5:
        raise RuntimeError(f"Checksum mismatch for {path}; delete it and retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/pdebench", help="Output directory.")
    parser.add_argument("--dataset", nargs="+", choices=sorted(FILES), default=sorted(FILES))
    args = parser.parse_args()

    for name in args.dataset:
        for entry in FILES[name]:
            download(args.root, *entry)
