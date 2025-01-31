import struct, array

import cv2
import numpy as np

# To open vectors for curious people
# Modified from: https://gist.github.com/twmht/f1d9c180e653739294ca5eb80851e344

def showvec(fn, width=24, height=24, resize=4.0):
    f = open(fn, 'rb')
    HEADERTYP = '<iihh'  # img count, img size, min, max

    # read header
    imgcount, imgsize, _, _ = struct.unpack(HEADERTYP, f.read(12))
    a = 0
    for i in range(imgcount):
        print(a)
        a += 1
        img = np.zeros((height, width), np.uint8)

        f.read(1)  # read gap byte

        data = array.array('h')

        ###  buf = f.read(imgsize*2)
        ###  data.fromstring(buf)

        data.fromfile(f, imgsize)

        for r in range(height):
            for c in range(width):
                img[r, c] = data[r * width + c]

        img = cv2.resize(img, (0, 0), fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('vec_img', img)
        k = 0xFF & cv2.waitKey(0)
        if k == 27:  # esc to exit
            break


showvec('wheel.vec')
