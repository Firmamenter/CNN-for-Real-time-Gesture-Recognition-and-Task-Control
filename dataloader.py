"""
Author: Da Chen
This script obtains all image names in the image folder and generate a data file which consists of image names and labels.
"""
import os
import glob

# Get all image names.
img_ok = glob.glob("./new_test_img/*ok*.png")
img_stop = glob.glob("./new_test_img/*stop*.png")
img_punch = glob.glob("./new_test_img/*punch*.png")
img_peace = glob.glob("./new_test_img/*peace*.png")
img_nothing = glob.glob("./new_test_img/*nothing*.png")

# Assigning labels.
img_ok = [name + ' 0' for name in img_ok]
img_stop = [name + ' 1' for name in img_stop]
img_punch = [name + ' 2' for name in img_punch]
img_peace = [name + ' 3' for name in img_peace]
img_nothing = [name + ' 4' for name in img_nothing]

# Create a data file by combining image names and labels.
if not os.path.isfile('./new_test.txt'):
    print 'Data file does not exist. Creating file ...'
    data_file = open('./new_test.txt', 'a')
    for item in img_ok:
        data_file.write("%s\n" % item)
    for item in img_stop:
        data_file.write("%s\n" % item)
    for item in img_punch:
        data_file.write("%s\n" % item)
    for item in img_peace:
        data_file.write("%s\n" % item)
    for item in img_nothing:
        data_file.write("%s\n" % item)
    data_file.close()
    print 'Done. Continue training ...'
else:
    print 'Data file exist. Continue training ...'
