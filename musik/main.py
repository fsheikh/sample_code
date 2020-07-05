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
from classify import DesiGenreDetector
from featext import AudioFeatureExtractor
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Links from eigene data-set
piya_say_naina = 'https://drive.google.com/uc?id=1oB3wFoZgHGE5gnP88KV505AcFrnEHfz_'
payar_akhan_de = 'https://drive.google.com/uc?id=1yBAY-NcxDPXQ__YPDDfAA7qPLza-zV8o'
khawaja = 'https://drive.google.com/uc?id=165I1Uf8LqaAiBm750NOxLbeDy8isJf3R'
zikar_parivash_ka = 'https://drive.google.com/uc?id=1L5VRFhhCF3oaQWyfynip_z_jVUQpqp07'
sikh_chaj = 'https://drive.google.com/uc?id=1PHVdbiAq_QDSAkbvV54WiNtFmxljROzd'
shikwa = 'https://drive.google.com/uc?id=16GVaIjXHwHWNHhPhGo4y6O5TwTxVJefx'
is_karam_ka = 'https://drive.google.com/uc?id=121UBXqQS0iwU5ZsznKyK4zWBEMcTf4FV'
mast_nazron_say = 'https://drive.google.com/uc?id=14uKXY_DTpVY5vx88HhBAz33azJofYXMd'
mohay_apnay_rang = 'https://drive.google.com/uc?id=1rrx6KSKnLzqy7095Wkac4doLdJ8r06H-'
meray_sohnaya = 'https://drive.google.com/uc?id=1Wjs2tO5X5GOJgI3nyGJ3RNjAl0ENEwXB'
mera_imaan_pak = 'https://drive.google.com/uc?id=1ebmOKPeAZ7Ouj8wJ9M3kflbTJEmXOd_e'
mujh_ko_teri_qasam = 'https://drive.google.com/uc?id=1R3QmRolu1mD4p_rYvNVsK8FflQlV01FA'
maye_ni_maye = 'https://drive.google.com/uc?id=18PITR8ZsTSaF4XuXALL5h9j8Tj4REJaW'
kise_da_yaar = 'https://drive.google.com/uc?id=1SWQ4Av4ck5Fy8XqX9lbv-f0MWGHD8iIL'
mere_saaqi = 'https://drive.google.com/uc?id=17NTxevTz827ClaR1ZISjzbMn9RZBYxoi'
ruthi_rut = 'https://drive.google.com/uc?id=1A-YO4pTd4u0rK-KrHOxTBjjP4ArfwpdJ'
rashk_e_qamar = 'https://drive.google.com/uc?id=17y9uvNrCG0kwSbv3alkH0XjdlkMf5zNC'
rab_nou_manana = 'https://drive.google.com/uc?id=1_1-bO2RtZBNu39zsWBtjtCsJozU-hX1u'
yt_rumi_qawali='https://www.youtube.com/watch?v=FjggJH45RRE'


# Map with song URLs and self-labelled genre-ground-truth (ggt)
# Second element of tuple is the Genre with following legend
# Q: Qawali
# G: Ghazal
# T: Thumri
# S: Song filmi or non-filmi geet
# F: Folk including dohay and kafian
ggt_map = {piya_say_naina : ('piya_say_naina.mp3', 'Q'),
    khawaja : ('khawaja.mp3', 'Q'),
    payar_akhan_de : ('payar_akhan_de.mp3', 'Q'),
    mujh_ko_teri_qasam : ('mujh_ko_teri_qasam.mp3', 'Q'),
    zikar_parivash_ka : ('zikar_parivash_ka.mp3', 'G'),
    shikwa : ('shikwa.mp3', 'Q'),
    is_karam_ka : ('is_karam_ka.mp3', 'Q'),
    mast_nazron_say : ('mast_nazron_say.mp3', 'Q'),
    mohay_apnay_rang : ('mohay_apnay_rang.mp3', 'Q'),
    sikh_chaj : ('sikh_chaj.mp3', 'Q'),
    meray_sohnaya : ('meray_sohnaya.mp3', 'S'),
    mera_imaan_pak : ('mera_imaan_pak.mp3', 'S'),
    maye_ni_maye : ('maye_ni_maye.mp3', 'F'),
    kise_da_yaar : ('kise_da_yaar.mp3', 'S'),
    mere_saaqi : ('mere_saaqi.mp3', 'Q'),
    ruthi_rut : ('ruthi_rut.mp3', 'S'),
    rab_nou_manana : ('ran_nou_manana.mp3', 'Q'),
    rashk_e_qamar : ('rashk_e_qamar.mp3', 'Q'),
    yt_rumi_qawali : ('rumi.mp3', 'Q')
}


# Smaller set for quick testing
#ggt_map = { khawaja : ('khawaja.mp3', 'Q'),
#            mera_imaan_pak : ('mera_imaan_pak.mp3', 'S'),
#            sikh_chaj : ('sikh_chaj.mp3', 'Q'),
#            ruthi_rut : ('ruthi_rut.mp3', 'G') }

if __name__ == "__main__":
    logger.info("Desi Music information retrieval: starting...")

    # TODO: separate false-negative, true-negative and false positive counters
    errorCount = 0.0
    totalQ = 0.0
    for songLink in ggt_map:
        song = AudioFeatureExtractor(ggt_map[songLink][0], songLink)
        q_features = song.extract_qawali_features()
        detector = DesiGenreDetector(q_features)
        totalQ = totalQ + 1.0
        if ggt_map[songLink][1] == 'Q':
            if detector.isQawali():
                logger.info("*** %s is correctly marked as a Qawali ***\n", ggt_map[songLink][0])
            else:
                errorCount = errorCount + 1.0
                logger.info("???Missed detecting=%s as Qawali???\n", ggt_map[songLink][0])
        else:
            if detector.isQawali():
                errorCount = errorCount + 1
                logger.info("%s is detected a Qawali but ground-truth=%s\n", ggt_map[songLink][0], ggt_map[songLink][1])
            else:
                logger.info("%s is corectly not marked as Qawali\n", ggt_map[songLink][0])

    logger.info("...Error rate=%6.4f percent...\n", 100 * (errorCount/totalQ))

