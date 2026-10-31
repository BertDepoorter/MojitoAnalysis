from mojito.download import download_brick
import os
from dotenv import load_dotenv

load_dotenv()
MY_PASSWORD = os.getenv("LISA_CONSORTIUM_KEY")
MY_USERNAME = os.getenv("LISA_CONSORTIUM_NAME")

for k in range(8):
    download_brick('emri', 
                               source_id=k, 
                               version='latest', 
                               reduced=True, 
                               cache_dir='/Users/bertd/Documents/PhD/LISA/data/MojitoLight', 
                               username=MY_USERNAME,
                               token=MY_PASSWORD)
