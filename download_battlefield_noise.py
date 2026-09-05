import os
import zipfile
import urllib.request
import shutil


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DOWNLOAD_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "downloads"
)

BATTLEFIELD_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "battlefield_noise"
)

NOISE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "noise"
)

# SESA dataset
SESA_URL = (
    "https://zenodo.org/records/3519845/files/"
    "SESA.zip?download=1"
)

ZIP_PATH = os.path.join(
    DOWNLOAD_DIR,
    "SESA.zip"
)


# ============================================================
# HOW MANY NEW FILES WE WANT
# ============================================================

# Keep the dataset reasonably small.
MAX_GUNSHOT = 30
MAX_EXPLOSION = 30


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(BATTLEFIELD_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

print("\n========================================")
print(" DOWNLOADING BATTLEFIELD SOUND DATA")
print("========================================")

if os.path.exists(ZIP_PATH):

    print("SESA.zip already exists.")
    print("Skipping download.")

else:

    print("Downloading SESA dataset...")
    print("Size: approximately 26 MB")

    try:

        urllib.request.urlretrieve(
            SESA_URL,
            ZIP_PATH
        )

        print("Download complete.")

    except Exception as e:

        print("\nERROR downloading dataset:")
        print(e)
        raise SystemExit(1)


# ============================================================
# EXTRACT ZIP TO TEMPORARY LOCATION
# ============================================================

EXTRACT_DIR = os.path.join(
    DOWNLOAD_DIR,
    "SESA_extracted"
)

os.makedirs(EXTRACT_DIR, exist_ok=True)

print("\nExtracting SESA dataset...")

try:

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zip_ref:

        zip_ref.extractall(EXTRACT_DIR)

except Exception as e:

    print("\nERROR extracting dataset:")
    print(e)
    raise SystemExit(1)

print("Extraction complete.")


# ============================================================
# FIND GUNSHOT / EXPLOSION FILES
# ============================================================

gunshot_files = []
explosion_files = []


for root, dirs, files in os.walk(EXTRACT_DIR):

    for filename in files:

        if not filename.lower().endswith(".wav"):
            continue

        full_path = os.path.join(
            root,
            filename
        )

        path_lower = full_path.lower()

        # Look for gunshot class
        if "gunshot" in path_lower:

            gunshot_files.append(full_path)

        # Look for explosion class
        elif "explosion" in path_lower:

            explosion_files.append(full_path)


print("\n========================================")
print(" FOUND BATTLEFIELD SOUNDS")
print("========================================")

print(
    f"Gunshot files found:   {len(gunshot_files)}"
)

print(
    f"Explosion files found: {len(explosion_files)}"
)


# ============================================================
# LIMIT DATASET SIZE
# ============================================================

gunshot_files = gunshot_files[:MAX_GUNSHOT]
explosion_files = explosion_files[:MAX_EXPLOSION]


# ============================================================
# COPY INTO BATTLEFIELD DIRECTORY
# ============================================================

print("\nCopying selected sounds...")

copied_gunshots = 0
copied_explosions = 0


def copy_unique(src, prefix, counter):

    filename = os.path.basename(src)

    destination_name = (
        f"{prefix}_{counter:03d}_{filename}"
    )

    destination = os.path.join(
        BATTLEFIELD_DIR,
        destination_name
    )

    # NEVER overwrite existing files
    if os.path.exists(destination):

        return False

    shutil.copy2(
        src,
        destination
    )

    return True


# Gunshots
for i, src in enumerate(
    gunshot_files,
    start=1
):

    if copy_unique(
        src,
        "gunshot",
        i
    ):

        copied_gunshots += 1


# Explosions
for i, src in enumerate(
    explosion_files,
    start=1
):

    if copy_unique(
        src,
        "explosion",
        i
    ):

        copied_explosions += 1


# ============================================================
# COPY INTO FINAL NOISE DIRECTORY
# ============================================================

print("\nAdding battlefield sounds to data/noise...")

copied_to_noise = 0
skipped_from_noise = 0


for filename in os.listdir(BATTLEFIELD_DIR):

    if not filename.lower().endswith(".wav"):
        continue

    src = os.path.join(
        BATTLEFIELD_DIR,
        filename
    )

    dst = os.path.join(
        NOISE_DIR,
        filename
    )

    # VERY IMPORTANT:
    # Never delete or overwrite existing noise.
    if os.path.exists(dst):

        skipped_from_noise += 1
        continue

    shutil.copy2(
        src,
        dst
    )

    copied_to_noise += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

existing_noise = [
    f
    for f in os.listdir(NOISE_DIR)
    if f.lower().endswith(".wav")
]

battlefield_noise = [
    f
    for f in os.listdir(BATTLEFIELD_DIR)
    if f.lower().endswith(".wav")
]


print("\n========================================")
print(" BATTLEFIELD DATA READY")
print("========================================")

print(
    f"New gunshots:          {copied_gunshots}"
)

print(
    f"New explosions:        {copied_explosions}"
)

print(
    f"Battlefield directory: {len(battlefield_noise)} WAV"
)

print(
    f"Total noise directory: {len(existing_noise)} WAV"
)

print(
    f"Location: {NOISE_DIR}"
)

print("\nExisting ESC-50 noise was NOT deleted.")

print("\nNext steps:")
print("1. python -m src.dataset_gen")
print("2. python -m src.train")
print("3. python -m src.test_model")