import os
import pandas as pd
from feature_extractor_func import YouTube8MFeatureExtractorFunc

if __name__ == '__main__':
  print('pwd = ' + str(os.getcwd()))
  apply_pca = True
  ve = YouTube8MFeatureExtractorFunc('D:/codeFile/pyFile/ve/yt8m-inception-pca/')
  a = pd.read_csv('test_data/v.csv')
  for i in range(a.shape[0]):
    i_inpath = a.loc[i, 'path']
    i_label = a.loc[i, 'label']
    i_outpath = 'test_data/' + str(i) + '_out.tfrecord'
    ve.video_extractor([i_inpath], [i_label], i_outpath, apply_pca)
    print('i_path = ' + str(i_inpath) + ', i_label = ' + str(i_label) + ', i_outpath = ' + str(i_outpath))







