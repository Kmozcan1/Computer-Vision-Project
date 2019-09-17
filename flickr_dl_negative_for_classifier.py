import cv2
import flickrapi
import urllib.request

# Downloads negative images for haar cascade classifier
# Modified from: https://gist.github.com/yunjey/14e3a069ad2aa3adf72dee93a53117d6

# API key
flickr = flickrapi.FlickrAPI('74a97a584f0debcfdaee61238997ccdd', 'cf81b4aecd570413', cache=True)

# Images to be downloaded
keyword = 'computer'
photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_m',
                     per_page=100,
                     sort='relevance')
urls = []

# number of images to be downloaded
number_of_images = 3999

for i, photo in enumerate(photos):

    url = photo.get('url_m')
    urls.append(url)
    if isinstance(url, str):
        print(str(i) + ': ' + url)
    if i > 3999:
        break

# Name of the image. Will be increased for each image.
# IMPORTANT: Change this with the +1 of last
# image number after starting to download the next set of images
image_name = 0
stop_when = 3999

for url in urls:
    # Stop when enough images are downloaded.
    # If desired, continue from the next image set by changing the
    # "image_name" and "keyword" parameters
    if image_name > stop_when:
        break
    image_name += 1
    if isinstance(url, str):
        name = str(image_name) + '.jpg'

        urllib.request.urlretrieve(url, "neg/" + name)
        img = cv2.imread("neg/" + name, cv2.IMREAD_GRAYSCALE)
        resizedImage = cv2.resize(img, (100, 100))
        cv2.imwrite("neg/" + name, resizedImage)

        # Resize the image and convert into greyscale
