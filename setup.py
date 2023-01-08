from pkg_resources import DistributionNotFound, get_distribution
from distutils.core import setup


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_deps = [
    'numpy',
    'regex',
    'tqdm',
    'gym'

]
tf_ver = '2.0.0a'
if get_dist('tensorflow>='+tf_ver) is None and get_dist('tensorflow_gpu>='+tf_ver) is None:
    install_deps.append('tensorflow>='+tf_ver)

setup(
  name = 'mitdeeplearning',         # How you named your package folder (MyLib)
  packages = ['mitdeeplearning'],   # Chose the same as "name"
  version = '0.3.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Official software labs for MIT Introduction to Deep Learning (http://introtodeeplearning.com)',   # Give a short description about your library
  author = 'Alexander Amini',                   # Type in your name
  author_email = 'introtodeeplearning-staff@mit.edu',      # Type in your E-Mail
  url = 'http://introtodeeplearning.com',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/aamini/introtodeeplearning/archive/v0.3.0.tar.gz',    # I explain this later on
  keywords = ['deep learning', 'neural networks', 'tensorflow', 'introduction'],   # Keywords that define your package best
  install_requires=install_deps,
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support',
    'Programming Language :: Python :: 3.6',
  ],
  package_data={
      'mitdeeplearning': ['bin/*', 'data/*', 'data/faces/DF/*', 'data/faces/DM/*', 'data/faces/LF/*', 'data/faces/LM/*'],
   },

)
