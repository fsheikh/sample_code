# Librosa client to extract and study features from
# custom data set (uploaded in personal google drive)
# Youtube on wishlist

# Setup
# =====
# Running with conda make sure proper environment is activated which contains the modules
# librosa, gdown, numpy and matplotlib
# With FMP installed on local machine conda activate FMP
# On a Linux machine, use pip to install above packages before running
# =====

import os
import logging
from featext import AudioFeatureExtractor
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Links from eigene data-set
piya_say_naina = 'https://drive.google.com/uc?id=1oB3wFoZgHGE5gnP88KV505AcFrnEHfz_'
khawaja = 'https://drive.google.com/uc?id=165I1Uf8LqaAiBm750NOxLbeDy8isJf3R'
zikar_parivash_ka = 'https://drive.google.com/uc?id=1L5VRFhhCF3oaQWyfynip_z_jVUQpqp07'
sikh_chaj = 'https://drive.google.com/uc?id=1PHVdbiAq_QDSAkbvV54WiNtFmxljROzd'
shikwa = 'https://drive.google.com/uc?id=16GVaIjXHwHWNHhPhGo4y6O5TwTxVJefx'
is_karam_ka = 'https://drive.google.com/uc?id=121UBXqQS0iwU5ZsznKyK4zWBEMcTf4FV'
mast_nazron_say = 'https://drive.google.com/uc?id=14uKXY_DTpVY5vx88HhBAz33azJofYXMd'
mohay_apnay_rang = 'https://drive.google.com/uc?id=1rrx6KSKnLzqy7095Wkac4doLdJ8r06H-'
meray_sohnaya = 'https://drive.google.com/uc?id=1Wjs2tO5X5GOJgI3nyGJ3RNjAl0ENEwXB'
mera_imaan_pak = 'https://drive.google.com/uc?id=1ebmOKPeAZ7Ouj8wJ9M3kflbTJEmXOd_e'
maye_ni_maye = 'https://drive.google.com/uc?id=18PITR8ZsTSaF4XuXALL5h9j8Tj4REJaW'
kise_da_yaar = 'https://drive.google.com/uc?id=1SWQ4Av4ck5Fy8XqX9lbv-f0MWGHD8iIL'
mere_saaqi = 'https://drive.google.com/uc?id=17NTxevTz827ClaR1ZISjzbMn9RZBYxoi'
ruthi_rut = 'https://drive.google.com/uc?id=1A-YO4pTd4u0rK-KrHOxTBjjP4ArfwpdJ'
rashk_e_qamar = 'https://drive.google.com/uc?id=17y9uvNrCG0kwSbv3alkH0XjdlkMf5zNC'
yt_rumi_qawali='https://www.youtube.com/watch?v=FjggJH45RRE'


url_map = {piya_say_naina : 'piya_say_naina.mp3',
    khawaja : 'khawaja.mp3',
    zikar_parivash_ka : 'zikar_parivash_ka.mp3',
    shikwa : 'shikwa.mp3',
    is_karam_ka : 'is_karam_ka.mp3',
    mast_nazron_say : 'mast_nazron_say.mp3',
    mohay_apnay_rang : 'mohay_apnay_rang.mp3',
    sikh_chaj : 'sikh_chaj.mp3',
    meray_sohnaya : 'meray_sohnaya.mp3',
    mera_imaan_pak : 'mera_imaan_pak.mp3',
    maye_ni_maye : 'maye_ni_maye.mp3',
    kise_da_yaar : 'kise_da_yaar.mp3',
    mere_saaqi : 'mere_saaqi.mp3',
    ruthi_rut : 'ruthi_rut.mp3',
    rashk_e_qamar : 'rashk_e_qamar.mp3',
    yt_rumi_qawali : 'rumi.mp3'
}

#url_map = {rashk_e_qamar : 'rashk_e_qamar.mp3' }
url_map = {yt_rumi_qawali : 'rumi.mp3' }
if __name__ == "__main__":
    logger.info("Feature extraction driver started")

    for songLink in url_map:
        song = AudioFeatureExtractor(url_map[songLink], songLink)
        song.extract_cqt()
        song.extract_beats()
        song.extract_mfcc_similarity()
    logger.info("Feature extraction done, check output directory for results!")
