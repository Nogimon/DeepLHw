from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

dire = '/media/zlab-1/Data/Lian/course/DeepLHw/coco/'
dataDir=dire + 'images'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dire,dataType)

coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]


# load and display image
#I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()


# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)


ann = anns[2]
polygons = []
for seg in ann['segmentation']:
    poly = np.array(seg).reshape((int(len(seg)/2), 2))
    polygons.append(Polygon(poly))
plt.plot(poly[:,0],poly[:,1])

for ann1 in anns:
    mask = coco.annToMask(ann1)
    plt.figure()
    plt.imshow(mask)


