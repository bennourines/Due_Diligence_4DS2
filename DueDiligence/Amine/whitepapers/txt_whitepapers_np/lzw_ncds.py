#!/usr/bin/env python3

import sys
import os
import pandas as pd
import io
import csv
import turbinator as turbokiller


crypto_data = pd.read_csv('/home/salva/Desktop/SpeedRun/Crypto/Crypto_dataset.csv')

files = []

crypto = list(crypto_data['Cryptocurrency'])
for f in crypto:
  files.append('/home/salva/Desktop/SpeedRun/Crypto/Whitepapers/txt_whitepapers_np/' + f + '.txt')

sizes = [os.path.getsize(f) for f in files]
contents = { f : io.FileIO(f).readall() for f in files}


def Z(contents):
  return turbokiller.compress(contents)

print("Compressing all files")
compressed = { f : Z(contents[f]) for f in files }

for f in crypto:
  print(f)
  filename = '/home/salva/Desktop/SpeedRun/Crypto/Whitepapers/txt_whitepapers_np/' + f + '.txt'
  print(compressed[filename])
  with open('/home/salva/Desktop/SpeedRun/Crypto/Diccionaris3/' + f + '.csv', 'w') as filehandle:
    for word in compressed[filename]:
      filehandle.write('%s\n' % str(word))
  filehandle.close()

