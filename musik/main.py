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

# Mixed desi songs with Links from eigene data-set (shared from Google drive)
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
# Qawali selection from youtube
yt_rumiQawali ='https://www.youtube.com/watch?v=FjggJH45RRE'
yt_dekhLayShakalMeri = 'https://www.youtube.com/watch?v=fG9tnmnQ7SM'
yt_tajdarHaram =  'https://www.youtube.com/watch?v=eFMLmCs19Gk'
yt_allahHoSabri = 'https://www.youtube.com/watch?v=vn2WuaDPJpE'
yt_mainSharabi = 'https://www.youtube.com/watch?v=8N9ZDsZZp58'
yt_shabEHijar = 'https://www.youtube.com/watch?v=aa8onCQryfQ'
yt_kuchIssAdaSay = 'https://www.youtube.com/watch?v=CKZhSNJn9vA'
yt_ajabAndaazTujhKo = 'https://www.youtube.com/watch?v=eXB7cYJtgOs'
yt_yehNaThiMeriQismat = 'https://www.youtube.com/watch?v=K43R5JAUOmE'
yt_meriDaastanHasrat = 'https://www.youtube.com/watch?v=VQUgqPOIa6Y'
yt_koiMilanBahanaSoch = 'https://www.youtube.com/watch?v=idNGyGVdQ2A'
yt_veekhanYaarTayParhan = 'https://www.youtube.com/watch?v=s_a4E-ywKmE'
yt_surkhAnkhonMain = 'https://www.youtube.com/watch?v=s4pFs0qWt6s'
yt_arabDeyMahi = 'https://www.youtube.com/watch?v=yWx9rP4yrCE'
yt_lagiValley = 'https://www.youtube.com/watch?v=0NhDOEPi6lU'
yt_makkiMadni = 'https://www.youtube.com/watch?v=gYZiFvbMu18'
yt_sahebTeriBandi = 'https://www.youtube.com/watch?v=1KQCcdFhA4k'
yt_azDairMughaAyam = 'https://www.youtube.com/watch?v=-R3OxAAlOx8'
yt_jalwaDildarDeedam = 'https://www.youtube.com/watch?v=Iz9hU6JjZXk'
yt_ghonghatChakSajna = 'https://www.youtube.com/watch?v=Ywa7cszhAUQ'
yt_mehfilMainBaarBaar = 'https://www.youtube.com/watch?v=khwmQ3PWpU0'
yt_awainTeTenuDasan = 'https://www.youtube.com/watch?v=P-Qe5yYLusU'
yt_udeekMainuSajna = 'https://www.youtube.com/watch?v=tbuOpgJOrH4'
yt_injVichrayMurrNayeAye = 'https://www.youtube.com/watch?v=4OmZe5Cv9kc'
yt_manKunToMaula = 'https://www.youtube.com/watch?v=Z5OWke4L-fE'

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
    rashk_e_qamar : ('rashk_e_qamar.mp3', 'Q')
}


# Smaller set for quick testing
small_set = { piya_say_naina : ('piya_say_naina.mp3', 'Q'),
            mera_imaan_pak : ('mera_imaan_pak.mp3', 'S'),
            sikh_chaj : ('sikh_chaj.mp3', 'Q'),
            ruthi_rut : ('ruthi_rut.mp3', 'G') }

# youtube qawali map
ytq_map = { yt_rumiQawali : ('rumi.mp3', 'Q'),
            #yt_yehNaThiMeriQismat : ('yeh_na_thi_hamari.mp3', 'Q'), # (uses other instruments)
            #yt_tajdarHaram : ('tajdar.mp3', 'Q'), # (very little taali, long alap in sample)
            yt_dekhLayShakalMeri : ('dekh_lsm.mp3' , 'Q'),
            #yt_allahHoSabri : ('allah_ho_sabri.mp3', 'Q'), # (included jhunjhuna)
            yt_mainSharabi : ('main_sharabi.mp3', 'Q'),
            yt_shabEHijar : ('shab_e_hijar.mp3', 'Q'),
            yt_kuchIssAdaSay : ('kuch_iss_ada.mp3', 'Q'),
            yt_ajabAndaazTujhKo : ('ajab_andaz.mp3', 'Q'),
            #yt_meriDaastanHasrat : ('meri_dastaan_hasrat.mp3', 'Q'), # (contains jhunjhuna)
            yt_koiMilanBahanaSoch : ('milan_bahana_soch.mp3', 'Q'),
            yt_veekhanYaarTayParhan : ('veekhan_yaar.mp3', 'Q'),
            yt_surkhAnkhonMain :('surkh_aankhon.mp3', 'Q'),
            #yt_arabDeyMahi : ('arab_dey_mahi.mp3', 'Q'), # (contains another instrument)
            #yt_lagiValley : ('lagi_valley.mp3', 'Q'),i #(contains bansari at the start shadowing harmonic profile)
            yt_makkiMadni : ('maki_madni.mp3', 'Q'),
            yt_sahebTeriBandi : ('saheb_teri_bandi.mp3', 'Q'),
            yt_azDairMughaAyam : ('az_dair_mugha.mp3', 'Q'),
            yt_jalwaDildarDeedam : ('jalwa_dildar_deedam.mp3', 'Q'),
            yt_ghonghatChakSajna : ('ghonghat_chak_sajna.mp3', 'Q'),
            yt_mehfilMainBaarBaar : ('mahfil_main_barbar.mp3', 'Q'),
            yt_awainTeTenuDasan : ('awayen_tay_tenu_dasan.mp3', 'Q'),
            #yt_udeekMainuSajna : ('udeek_mainu_sajna.mp3','Q'), # (suspected other instruments)
            yt_injVichrayMurrNayeAye : ('inj_vichray_murr.mp3', 'Q'),
            yt_manKunToMaula : ('man_kun_tow_maula.mp3', 'Q')
}

# Runs feature extractor and genre detection loop on the given dataset.
# Returns a list with total items processed, true-negatives, false-negatives and
# false positive
def genre_errors(dataset, genre='Q'):
    # Total, false-negative, false-positive, true-negative, true-positive
    counterList = [0.0, 0.0, 0.0, 0.0, 0.0]
    for song in dataset:
        songData = AudioFeatureExtractor(dataset[song][0], song)
        songFeatures = songData.extract_qawali_features()
        songDetector = DesiGenreDetector(songFeatures)
        counterList[0] = counterList[0] + 1.0
        if dataset[song][1] == genre and songDetector.isQawali():
            logger.info("*** %s is correctly marked as a Qawali ***\n", dataset[song][0])
            counterList[4] = counterList[4] + 1.0
        elif dataset[song][1] == genre:
            logger.info("??? =%s is Qawali but detection missed ???\n", dataset[song][0])
            counterList[1] = counterList[1] + 1.0
        elif songDetector.isQawali():
            logger.info("!!! %s is detected a Qawali but ground-truth=%s !!!\n", dataset[song][0], dataset[song][1])
            counterList[2] = counterList[2] + 1.0
        else:
            logger.info('Non-Qawali=%s correctly marked', dataset[song][0])
            counterList[3] = counterList[3] + 1.0

        if counterList[0] != counterList[1] + counterList[2] + counterList[3] + counterList[4]:
            logger.error("Error statistics don't add up correctly")
            return [-1.0, -1.0, -1.0, -1.0, -1.0]

    return counterList


if __name__ == "__main__":
    logger.info("\n\nDesi Music information retrieval: starting...\n\n")

    RunSet = "full"

    if RunSet == 'small':
        ss_stats = genre_errors(small_set)
        total = ss_stats[0]
        fails = ss_stats[1] + ss_stats[2]
        success = ss_stats[3] + ss_stats[4]
        s_rate = 100 * (success/total)
        f_rate = 100 * (fails/total)
        logger.info("\nResults small-set\n")
        logger.info("\n-----------------\n")
        logger.info("Total = %6.4f Errors = %6.4f Error rate=%6.4f percent...\n", total, fails, f_rate)

    if RunSet == "full":
        fs_stats = genre_errors(ggt_map)
        total = fs_stats[0]
        fails = fs_stats[1] + fs_stats[2]
        success = fs_stats[3] + fs_stats[4]
        s_rate = 100 * (success/total)
        f_rate = 100 * (fails/total)
        logger.info("\nResults full-set\n")
        logger.info("\n----------------\n")
        logger.info("Total = %6.4f Errors = %6.4f Error rate=%6.4f percent...\n", total, fails, f_rate)

    if RunSet == "youtube":
        yt_stats = genre_errors(ytq_map)
        total = yt_stats[0]
        fails = yt_stats[1] + yt_stats[2]
        success = yt_stats[3] + yt_stats[4]
        s_rate = 100 * (success/total)
        f_rate = 100 * (fails/total)
        logger.info("\nResults youtube-set\n")
        logger.info("\n-------------------\n")
        logger.info("Total = %6.4f Errors = %6.4f Error rate=%6.4f percent...\n", total, fails, f_rate)

