"""Download the Mojito-Light EMRI bricks, the instrument-noise brick and the EMRI catalog.

Credentials are read from .env (LISA_CONSORTIUM_NAME, LISA_CONSORTIUM_KEY).
Usage: python fetch_data.py [--sources 0 1 ... 7] [--cache-dir DIR]
"""
import argparse
import os

from dotenv import load_dotenv
from mojito.download import download_brick, download_catalog

load_dotenv()
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--sources', type=int, nargs='+', default=list(range(8)))
ap.add_argument('--cache-dir', default='/Users/bertd/Documents/PhD/LISA/data/MojitoLight')
args = ap.parse_args()
auth = dict(cache_dir=args.cache_dir, username=os.getenv('LISA_CONSORTIUM_NAME'), token=os.getenv('LISA_CONSORTIUM_KEY'))

for k in args.sources:
    print('emri', k, download_brick('emri', source_id=k, version='latest', reduced=True, **auth))
print('noise', download_brick('noise', version='latest', reduced=True, **auth))
print('catalog', download_catalog('emri', version='latest', **auth))
