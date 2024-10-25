from src.nn_models.segnet import SegNet
import os
from skimage.io import imread
from skimage.transform import resize
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image

from src.data.provider.gdrive_weights import WeightsProvider
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

weights_provider= WeightsProvider(
        file_id=os.getenv("SEGNET_WEIGHTS_GID"),
        weights_path = Path(os.getenv("WEIGHTS_PATH_ROOT", "./artifacts/models"))
    )
segnet = SegNet(
    weights_provider=weights_provider
)
device = "cpu"
images = []
lesions = []

root = "./dataset/PH2Dataset"
# segnet.load_state_dict(
#     torch.load(
#         "./artifacts/models/SegNet/SegNet_dice_100e_ad.pt",
#         map_location=torch.device('cpu'),
#     )
# )
weights_path = weights_provider.get_weights_path()

segnet.load_from_provider(
    weights_name=os.getenv("SEGNET_WEIGHTS_NAME")
)
segnet.eval()

for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset images')):
    if root.endswith('_Dermoscopic_Image'):
        images.append(imread(os.path.join(root, files[0])))
    if root.endswith('_lesion'):
        lesions.append(imread(os.path.join(root, files[0])))

size = (256, 256)
X = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images]
Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]


img = torch.from_numpy(np.rollaxis(X[4][np.newaxis, :], 3, 1)).to(torch.device('cpu'), dtype=torch.float32)

print(X[4][0, :10])

print(img.shape)

inference = segnet(img)
inference = inference.detach().numpy()

# matplotlib.image.imsave('image.png', np.rollaxis(img.numpy(), 1, 4).squeeze(0))
print(inference.shape)


plt.figure()

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,1)

axarr[0].imshow(np.rollaxis(img.numpy(), 1, 4).squeeze(0))
axarr[1].imshow(np.rollaxis(inference, 1, 4).squeeze(0))

print(inference[0, 0, : 10])

plt.show()
