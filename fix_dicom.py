import os
import pydicom
import SimpleITK as sitk
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP


# --- CONFIG ---
dicom_dir = "4100844 PCCT 4D-CTA/00000002"
output_4d_path = "output_cta_4d.nii.gz"

all_slices = []
for fname in sorted(os.listdir(dicom_dir)):
    if fname.lower().endswith(".dcm"):
        path = os.path.join(dicom_dir, fname)
        ds = pydicom.dcmread(path, stop_before_pixels=False)
        all_slices.append((ds.InstanceNumber, path))

# Sort by InstanceNumber
all_slices.sort()

# Group every 139 slices
frames = []
slices_per_frame = 139
for i in range(0, len(all_slices), slices_per_frame):
    group = all_slices[i:i + slices_per_frame]
    if len(group) < slices_per_frame:
        continue  # incomplete frame
    paths = [p for _, p in group]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    vol = reader.Execute()
    frames.append(vol)
    print(f"Saved frame {len(frames)-1} with {len(paths)} slices")

# # --- STEP 3: Stack all frames into 4D volume ---
# volume_4d = sitk.JoinSeries(frames)
# volume_4d.SetSpacing((*frames[0].GetSpacing(), 1.0))  # Last dim = time
#
# # --- STEP 4: Save as .nii.gz ---
# sitk.WriteImage(volume_4d, output_4d_path)
# print(f"Saved 4D CTA to: {output_4d_path}")

# Same as seperate frames
for i, vol in enumerate(frames):
    sitk.WriteImage(vol, f"4100844_frames/frame_{i}.nii.gz")
    print(f"Saved frame {i}")
