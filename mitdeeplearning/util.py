import matplotlib.pyplot as plt
import tensorflow as tf
import time
import progressbar

from IPython import display as ipythondisplay
from string import Formatter




#####################################
def custom_progress_text(message):

  message_ = message.replace('(', '{')
  message_ = message_.replace(')', '}')

  keys = [key[1] for key in Formatter().parse(message_)]

  ids = {}
  for key in keys:
    if key is not None:
      ids[key] = float('nan')

  msg = progressbar.FormatCustomText(message, ids)
  return msg

def create_progress_bar(text=None):
  if text is None:
    text = progressbar.FormatCustomText('')
  bar = progressbar.ProgressBar(widgets=[
      progressbar.Percentage(),
      progressbar.Bar(),
      progressbar.AdaptiveETA(), '  ',
      text,
  ])
  return bar

def display_model(model):
  tf.keras.utils.plot_model(model,
             to_file='tmp.png',
             show_shapes=True)
  return ipythondisplay.Image('tmp.png')


class LossHistory:
  def __init__(self, smoothing_factor=0.0):
    self.alpha = smoothing_factor
    self.loss = []
  def append(self, value):
    self.loss.append( self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss)>0 else value )
  def get(self):
    return self.loss

class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        plt.plot(data)
      elif self.scale == 'semilogx':
        plt.semilogx(data)
      elif self.scale == 'semilogy':
        plt.semilogy(data)
      elif self.scale == 'loglog':
        plt.loglog(data)
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
      ipythondisplay.clear_output(wait=True)
      ipythondisplay.display(plt.gcf())

      self.tic = time.time()
