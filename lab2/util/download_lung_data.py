from urllib2 import urlopen
import os

urls = {'pneumothorax_test':'https://www.dropbox.com/s/x74ykyivipwnozs/pneumothorax_test.h5?dl=1',
        'pneumothorax_train':'https://www.dropbox.com/s/pnwf67qzztd1slc/pneumothorax_train.h5?dl=1'}

data_dir = 'lung_data/'

for (name,url) in urls.iteritems():
    if not os.path.isfile(data_dir+name+'.h5'):
        print('Downloading '+name+'...')
        u = urlopen(url)
        data = u.read()
        u.close()

        with open(data_dir+name+'.h5', "wb") as f :
            f.write(data)
print('Files have been downloaded.')