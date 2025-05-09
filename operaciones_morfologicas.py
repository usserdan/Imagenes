import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
# Leer la imagen
image = sitk.ReadImage(r"segmentations/MRBrainTumor1 mask.nrrd", sitk.sitkUInt8)

# Convertir la imagen a array de NumPy para visualización
image_np = sitk.GetArrayFromImage(image)

# Elegir un corte axial
voxel_counts = np.sum(image_np, axis=(1, 2))
slice_idx = np.argmax(voxel_counts)  # Corte con más señal

# ----------------- EROSIÓN -----------------
# Aplicar erosión binaria
eroded = sitk.BinaryErode(
    image,
    kernelRadius=[1, 1, 1],  # Radio del elemento estructurante
    kernelType=sitk.sitkBall,  # Forma del kernel
    backgroundValue=0,
    foregroundValue=1,
    boundaryToForeground=True
)

# ----------------- DILATACIÓN -----------------
# Aplicar dilatación binaria
dilated = sitk.BinaryDilate(
    image,
    kernelRadius=[1, 1, 1],  # Radio del elemento estructurante
    kernelType=sitk.sitkBall,  # Forma del kernel
    backgroundValue=0,
    foregroundValue=1,
    boundaryToForeground=True
)

# ----------------- APERTURA (Erosión + Dilatación) -----------------
# Aplicar erosión binaria
eroded_open = sitk.BinaryErode(
    image,
    kernelRadius=[1, 1, 1],
    kernelType=sitk.sitkBall,
    backgroundValue=0,
    foregroundValue=1,
    boundaryToForeground=True
)

# Aplicar dilatación sobre la imagen erosionada
opened = sitk.BinaryDilate(
    eroded_open,
    kernelRadius=[1, 1, 1],
    kernelType=sitk.sitkBall,
    backgroundValue=0,
    foregroundValue=1,
    boundaryToForeground=True
)

# ----------------- CIERRE (Dilatación + Erosión) -----------------
# Aplicar dilatación binaria
dilated_close = sitk.BinaryDilate(
    image,
    kernelRadius=[1, 1, 1],
    kernelType=sitk.sitkBall,
    backgroundValue=0,
    foregroundValue=1,
    boundaryToForeground=True
)

# Aplicar erosión sobre la imagen dilatada
closed = sitk.BinaryErode(
    dilated_close,
    kernelRadius=[1, 1, 1],
    kernelType=sitk.sitkBall,
    backgroundValue=0,
    foregroundValue=1,
    boundaryToForeground=True
)

eroded_np = sitk.GetArrayFromImage(eroded)
dilated_np = sitk.GetArrayFromImage(dilated)
opened_np = sitk.GetArrayFromImage(opened)
closed_np = sitk.GetArrayFromImage(closed)

# Crear figura con 4 filas (una por operación) y 2 columnas (resultado y superposición)
fig, axs = plt.subplots(4, 2, figsize=(12, 16))
axs = axs.reshape(4, 2)

# ----------- EROSION -----------
axs[0, 0].imshow(eroded_np[slice_idx], cmap='gray')
axs[0, 0].set_title("Erosión")
axs[0, 0].axis("off")

axs[0, 1].imshow(image_np[slice_idx], cmap='gray')
axs[0, 1].imshow(eroded_np[slice_idx], cmap='Reds', alpha=0.5)
axs[0, 1].set_title("Original + Erosión")
axs[0, 1].axis("off")

# ----------- DILATACIÓN -----------
axs[1, 0].imshow(dilated_np[slice_idx], cmap='gray')
axs[1, 0].set_title("Dilatación")
axs[1, 0].axis("off")

axs[1, 1].imshow(image_np[slice_idx], cmap='gray')
axs[1, 1].imshow(dilated_np[slice_idx], cmap='Reds', alpha=0.5)
axs[1, 1].set_title("Original + Dilatación")
axs[1, 1].axis("off")

# ----------- APERTURA -----------
axs[2, 0].imshow(opened_np[slice_idx], cmap='gray')
axs[2, 0].set_title("Apertura")
axs[2, 0].axis("off")

axs[2, 1].imshow(image_np[slice_idx], cmap='gray')
axs[2, 1].imshow(opened_np[slice_idx], cmap='Reds', alpha=0.5)
axs[2, 1].set_title("Original + Apertura")
axs[2, 1].axis("off")

# ----------- CIERRE -----------
axs[3, 0].imshow(closed_np[slice_idx], cmap='gray')
axs[3, 0].set_title("Cierre")
axs[3, 0].axis("off")

axs[3, 1].imshow(image_np[slice_idx], cmap='gray')
axs[3, 1].imshow(closed_np[slice_idx], cmap='Reds', alpha=0.5)
axs[3, 1].set_title("Original + Cierre")
axs[3, 1].axis("off")

plt.tight_layout()
plt.show()

# Ruta donde guardar las segmentaciones
output_dir = r"segmentations/ouputs"
os.makedirs(output_dir, exist_ok=True)  # Crea la carpeta si no existe

# Asegurarse de que solo haya 0s y 1s
eroded = eroded > 0
dilated = dilated > 0
opened = opened > 0
closed = closed > 0

# Guardar las segmentaciones en formato .nrrd
sitk.WriteImage(eroded, os.path.join(output_dir, "eroded.nrrd"))
sitk.WriteImage(dilated, os.path.join(output_dir, "dilated.nrrd"))
sitk.WriteImage(opened, os.path.join(output_dir, "opened.nrrd"))
sitk.WriteImage(closed, os.path.join(output_dir, "closed.nrrd"))