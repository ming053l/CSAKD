import scipy
import sklearn
from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
images = loadmat('H01_56.mat',variable_names='IMAGES',appendmat=True).get('IMAGES')

imgplot = plt.imshow(images[:,:,0])

plt.show()
