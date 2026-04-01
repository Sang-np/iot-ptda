import cv2 as cv
import numpy as np
import urllib.request

def read_img_url(url):
    req = urllib.request.urlopen(url)
    img_rw = np.asarray( bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(img_rw, 3)
    return img

if __name__=="__main__":
    url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/samples/data/lena.jpg"
    img = read_img_url(url)
    cv.imshow("img",img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    n_ = add_noise(img)
    cv.imshow("img", n_)
    cv.waitKey(0)
    cv.destroyAllWindows()
    imt = img.copy()
    img2 = np.clip(imt + n_, 0, 255).astype(np.uint8)
    cv.imshow("img",img2)
    cv.waitKey(0)
    cv.destroyAllwindows()
    
    cb = np.concatenate((img, n_, img2), axis=1)
    cv.imshow("img", cb)
    cv.waitKey(0)
    cv.destroyAllWindows()