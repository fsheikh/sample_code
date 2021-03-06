# Driver for supervised learning of Qawali genre

# Copyright (C) 2020  Faheem Sheikh (fahim.sheikh@gmail.com)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>

# Uses self-labelled qawalis sources from a personal collection
# and also publically available e.g. via youtube. Selection criteria
# for training data is begining contains tabla/taali as well as harmonimum
# outlining main melody/raag

# Dependencies
# ============
# pytorch, librosa, gdown, numpy and matplotlib


import os
import logging
from classify import DesiGenreDetector
from featext import AudioFeatureExtractor
from gtzan import GtzanMap
from qlearner import QawaliClassifier
from qlearner import LoadMode
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# URLs for training data used to teach a neural network
piya_say_naina = 'https://drive.google.com/uc?id=1oB3wFoZgHGE5gnP88KV505AcFrnEHfz_'
khawaja = 'https://drive.google.com/uc?id=165I1Uf8LqaAiBm750NOxLbeDy8isJf3R'
is_karam_ka = 'https://drive.google.com/uc?id=121UBXqQS0iwU5ZsznKyK4zWBEMcTf4FV'
mohay_apnay_rang = 'https://drive.google.com/uc?id=1rrx6KSKnLzqy7095Wkac4doLdJ8r06H-'
mere_saaqi = 'https://drive.google.com/uc?id=17NTxevTz827ClaR1ZISjzbMn9RZBYxoi'
yt_kuchIssAdaSay = 'https://www.youtube.com/watch?v=CKZhSNJn9vA'
yt_veekhanYaarTayParhan = 'https://www.youtube.com/watch?v=s_a4E-ywKmE'
yt_aamadaBaQatal = 'https://www.youtube.com/watch?v=Vfs0_cPcOtg'
yt_nerreNerreVass = 'https://www.youtube.com/watch?v=WnPYEjOOc0A'
yt_ajabAndaazTujhKo = 'https://www.youtube.com/watch?v=eXB7cYJtgOs'

meray_sohnaya = 'https://drive.google.com/uc?id=1Wjs2tO5X5GOJgI3nyGJ3RNjAl0ENEwXB'
mera_imaan_pak = 'https://drive.google.com/uc?id=1ebmOKPeAZ7Ouj8wJ9M3kflbTJEmXOd_e'
maye_ni_maye = 'https://drive.google.com/uc?id=18PITR8ZsTSaF4XuXALL5h9j8Tj4REJaW'
kise_da_yaar = 'https://drive.google.com/uc?id=1SWQ4Av4ck5Fy8XqX9lbv-f0MWGHD8iIL'
ruthi_rut = 'https://drive.google.com/uc?id=1A-YO4pTd4u0rK-KrHOxTBjjP4ArfwpdJ'
zikar_parivash_ka = 'https://drive.google.com/uc?id=1L5VRFhhCF3oaQWyfynip_z_jVUQpqp07'
maye_ni_kinnon_akhan = 'https://drive.google.com/uc?id=1KXb6v6cuHq2nfPcFmPxcMCGBhC5QlgLk'
yaar_dhaadi = 'https://drive.google.com/uc?id=1sMC9CbVGdT9jb1vBYUe_nXUzWotBMNFJ'
nahi_lagay_jiya = 'https://drive.google.com/uc?id=1z2InTLyjPU2d5Z-sRAmNDu8oqs-o6nQK'
yt_naqsh_khayal = 'https://www.youtube.com/watch?v=uFfMzXMogAQ'

# URLs for test data
rashk_e_qamar = 'https://drive.google.com/uc?id=17y9uvNrCG0kwSbv3alkH0XjdlkMf5zNC'
rab_nou_manana = 'https://drive.google.com/uc?id=1_1-bO2RtZBNu39zsWBtjtCsJozU-hX1u'
mujh_ko_teri_qasam = 'https://drive.google.com/uc?id=1R3QmRolu1mD4p_rYvNVsK8FflQlV01FA'
mast_nazron_say = 'https://drive.google.com/uc?id=14uKXY_DTpVY5vx88HhBAz33azJofYXMd'
yt_rumiQawali ='https://www.youtube.com/watch?v=FjggJH45RRE'
yt_dekhLayShakalMeri = 'https://www.youtube.com/watch?v=fG9tnmnQ7SM'
yt_tajdarHaram =  'https://www.youtube.com/watch?v=eFMLmCs19Gk'
yt_allahHoSabri = 'https://www.youtube.com/watch?v=vn2WuaDPJpE'
yt_mainSharabi = 'https://www.youtube.com/watch?v=8N9ZDsZZp58'
yt_shabEHijar = 'https://www.youtube.com/watch?v=aa8onCQryfQ'
yt_yehNaThiMeriQismat = 'https://www.youtube.com/watch?v=K43R5JAUOmE'
yt_meriDaastanHasrat = 'https://www.youtube.com/watch?v=VQUgqPOIa6Y'
yt_koiMilanBahanaSoch = 'https://www.youtube.com/watch?v=idNGyGVdQ2A'
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
yt_kaanRozkeRooh = 'https://www.youtube.com/watch?v=CK1Ypt3dOUs'
yt_aeyMairayHamnashin = 'https://www.youtube.com/watch?v=aWAxus8MfsM'
yt_tumheDilLagi = 'https://www.youtube.com/watch?v=K9kAI20osxc'
yt_laalMeriPat = 'https://www.youtube.com/watch?v=SihdsEzawaU&t=75s'
yt_guftamKayRoshan = 'https://www.youtube.com/watch?v=NB6ZS6US-hc'
yt_tuKareemiMun = 'https://www.youtube.com/watch?v=EAzXFx_9dGc'
yt_harLehzaBashakal = 'https://www.youtube.com/watch?v=4mJzU3fhJjY'
# Map with song URLs and self-labelled genre-ground-truth (ggt)
# Second element of tuple is the Genre with following legend
# Q: Qawali
# G: Ghazal
# T: Thumri
# S: Song filmi or non-filmi geet
# F: Folk including dohay and kafian

# Training data chosen with five performances from NFAK in Qawali genre
# followed by five other samples each by a different qawal group.
# Non-Qawali items are a mix of pop, folk and Ghazals including some items
# from NFAK to reduce the bias
# Training data is then extended with 10 songs of each genre from GZTAN dataset
# Compensation for too little training data for Qawalis, each Qawali in training
# set is repeated four times with each time feature extraction done via a time
# offset of 30 seconds
# FIXME: Looking at the graphs produces of CQT by offset versions of same songs,
# will not be useable to detect the tabla-taali combination since the later on
# segments look similar to other genere without the characteristic high energy
# in middle/low pitches.
tarbiati_set = {piya_say_naina : ('piya_say_naina.mp3', 'Q', 0.0),
    #'piya_say_naina_30' : ('piya_say_naina.mp3', 'Q', 30.0),
    #'piya_say_naina_60' : ('piya_say_naina.mp3', 'Q', 60.0),
    #'piya_say_naina_90' : ('piya_say_naina.mp3', 'Q', 90.0),
    khawaja : ('khawaja.mp3', 'Q', 0.0),
    #'khawaja_30' : ('khawaja.mp3', 'Q', 30.0),
    #'khawaja_60' : ('khawaja.mp3', 'Q', 60.0),
    #'khawaja_90' : ('khawaja.mp3', 'Q', 90.0),
    is_karam_ka : ('is_karam_ka.mp3', 'Q', 0.0),
    #'is_karam_ka_30' : ('is_karam_ka.mp3', 'Q', 30.0),
    #'is_karam_ka_60' : ('is_karam_ka.mp3', 'Q', 60.0),
    #'is_karam_ka_90' : ('is_karam_ka.mp3', 'Q', 90.0),
    mohay_apnay_rang : ('mohay_apnay_rang.mp3', 'Q', 0.0),
    #'mohay_apnay_rang_30' : ('mohay_apnay_rang.mp3', 'Q', 30.0),
    #'mohay_apnay_rang_60' : ('mohay_apnay_rang.mp3', 'Q', 60.0),
    #'mohay_apnay_rang_90' : ('mohay_apnay_rang.mp3', 'Q', 90.0),
    mere_saaqi : ('mere_saaqi.mp3', 'Q', 0.0),
    #'mere_saaqi_30' : ('mere_saaqi.mp3', 'Q', 30.0),
    #'mere_saaqi_60' : ('mere_saaqi.mp3', 'Q', 60.0),
    #'mere_saaqi_90' : ('mere_saaqi.mp3', 'Q', 90.0),
    meray_sohnaya : ('meray_sohnaya.mp3', 'S', 0.0),
    mera_imaan_pak : ('mera_imaan_pak.mp3', 'S', 0.0),
    maye_ni_maye : ('maye_ni_maye.mp3', 'F', 0.0),
    kise_da_yaar : ('kise_da_yaar.mp3', 'S', 0.0),
    yt_kuchIssAdaSay : ('kuch_iss_ada.mp3', 'Q', 0.0),
    #'yt_kuchIssAdaSay_30' : ('kuch_iss_ada.mp3', 'Q', 30.0),
    #'yt_kuchIssAdaSay_60' : ('kuch_iss_ada.mp3', 'Q', 60.0),
    #'yt_kuchIssAdaSay_90' : ('kuch_iss_ada.mp3', 'Q', 90.0),
    yt_veekhanYaarTayParhan : ('veekhan_yaar.mp3', 'Q', 0.0),
    #'yt_veekhanYaarTayParhan_30' : ('veekhan_yaar.mp3', 'Q', 30.0),
    #'yt_veekhanYaarTayParhan_60' : ('veekhan_yaar.mp3', 'Q', 60.0),
    #'yt_veekhanYaarTayParhan_90' : ('veekhan_yaar.mp3', 'Q', 90.0),
    yt_aamadaBaQatal : ('aamada_ba_qatal.mp3', 'Q', 0.0),
    #'yt_aamadaBaQatal_30' : ('aamada_ba_qatal.mp3', 'Q', 30.0),
    #'yt_aamadaBaQatal_60' : ('aamada_ba_qatal.mp3', 'Q', 60.0),
    #'yt_aamadaBaQatal_90' : ('aamada_ba_qatal.mp3', 'Q', 90.0),
    yt_nerreNerreVass : ('nerre_nerre_vass.mp3', 'Q', 0.0),
    #'yt_nerreNerreVass_30' : ('nerre_nerre_vass.mp3', 'Q', 30.0),
    #'yt_nerreNerreVass_60' : ('nerre_nerre_vass.mp3', 'Q', 60.0),
    #'yt_nerreNerreVass_90' : ('nerre_nerre_vass.mp3', 'Q', 90.0),
    yt_ajabAndaazTujhKo : ('ajab_andaz.mp3', 'Q', 0.0),
    #'yt_ajabAndaazTujhKo_30' : ('ajab_andaz.mp3', 'Q', 30.0),
    #'yt_ajabAndaazTujhKo_60' : ('ajab_andaz.mp3', 'Q', 60.0),
    #'yt_ajabAndaazTujhKo_90' : ('ajab_andaz.mp3', 'Q', 90.0),
    ruthi_rut : ('ruthi_rut.mp3', 'S', 0.0),
    zikar_parivash_ka : ('zikar_parivash_ka.mp3', 'G', 0.0),
    maye_ni_kinnon_akhan : ('maye_ni_kinnon_akhan.mp3', 'F', 0.0),
    yaar_dhaadi : ('yaar_dhaadi.mp3', 'F', 0.0),
    nahi_lagay_jiya : ('nahi_lagay_jiya.mp3', 'T', 0.0),
    yt_naqsh_khayal : ('naqsh_khayal.mp3', 'G', 0.0)
}

#training_data = {piya_say_naina : ('piya_say_naina.mp3', 'Q'),
#    khawaja : ('khawaja.mp3', 'Q')}

# Songs to be used for testing
imtayhani_set = { yt_rumiQawali : ('rumi.mp3', 'Q'),
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
            'gtzan_pop.00005' : ('reggae/reggae.00005.au', 'po'),
            'gtzan_pop.00015' : ('reggae/reggae.00015.au', 'po'),
            'gtzan_pop.00025' : ('reggae/reggae.00025.au', 'po'),
            'gtzan_pop.00035' : ('reggae/reggae.00035.au', 'po'),
            yt_sahebTeriBandi : ('saheb_teri_bandi.mp3', 'Q'),
            yt_azDairMughaAyam : ('az_dair_mugha.mp3', 'Q'),
            yt_jalwaDildarDeedam : ('jalwa_dildar_deedam.mp3', 'Q'),
            yt_ghonghatChakSajna : ('ghonghat_chak_sajna.mp3', 'Q'),
            yt_mehfilMainBaarBaar : ('mahfil_main_barbar.mp3', 'Q'),
            yt_awainTeTenuDasan : ('awayen_tay_tenu_dasan.mp3', 'Q'),
            #yt_udeekMainuSajna : ('udeek_mainu_sajna.mp3','Q'), # (suspected other instruments)
            yt_injVichrayMurrNayeAye : ('inj_vichray_murr.mp3', 'Q'),
            yt_manKunToMaula : ('man_kun_tow_maula.mp3', 'Q'),
            yt_kaanRozkeRooh : ('kaan_roz_ke_rooh.mp3', 'Q'),
            yt_aeyMairayHamnashin : ('aey_mairay_hamnashin.mp3', 'Q'),
            yt_tumheDilLagi : ('tumhe_dil_lagi.mp3', 'Q'),
            yt_laalMeriPat : ('laal_meri_pat.mp3', 'Q'),
            yt_aamadaBaQatal : ('aamada_ba_qatal.mp3', 'Q'),
            yt_harLehzaBashakal : ('har_lehza_bashakal.mp3', 'Q'),
            yt_tuKareemiMun : ('tu_kareemi_mun.mp3', 'Q')
}

if __name__ == "__main__":
    logger.info("\n\n Supervised Qawali Learning...\n\n")
    training_data = {}
    gtzan_train = GtzanMap('/home/fsheikh/musik/gtzan/genres')
    for genre in GtzanMap.Genres:
        g_map = gtzan_train.cmap(genre, 10)
        training_data.update(g_map)

    training_data.update(tarbiati_set)
    # Feature vectors are a function of time, each vector contains pitch energy per Midi/frequency
    # and spectral energy per audio subband, averaged over an observation window
    # parameter in extract feature
    #N = AudioFeatureExtractor.FreqBins + AudioFeatureExtractor.SubBands
    N = AudioFeatureExtractor.FreqBins
    T = len(training_data)
    E = len(imtayhani_set)
    logger.info("Training with data elements=%d and features=%d", T, N)
    qc = QawaliClassifier(T,E,N)

    Phase = "ReloadAndRun"
    if Phase == "Loading":
        # Load feature data and save to disk for later processing
        # avoiding the I/O intensive load proceedure
        for song in training_data:
            offset = 0.0 if (len(training_data[song]) == 2) else training_data[song][2]
            songData = AudioFeatureExtractor(training_data[song][0], song, offset)
            songFeatures = songData.extract_qawali_features()
            # Input parameters are features and genre
            qc.load_one(songFeatures, training_data[song][1])

        qc.load_complete(LoadMode.TI)
        for song in imtayhani_set:
            songData = AudioFeatureExtractor(imtayhani_set[song][0], song)
            songFeatures = songData.extract_qawali_features()
            # Input parameters are features and genre
            qc.load_one(songFeatures, imtayhani_set[song][1], LoadMode.EI)
        qc.load_complete(LoadMode.EI)
        qc.save_and_plot()
    elif Phase == "ReloadAndRun":
        #tmp_song = AudioFeatureExtractor(tarbiati_set[kise_da_yaar][0], kise_da_yaar)
        #tmp_features = tmp_song.extract_qawali_features()
        qc.reload_from_disk()
        for epoch in range(1):
            logger.info("\nTraining for epoch=%d\n", epoch)
            qc.train()

        qc.classify()
    else:
        logger.error("Invalid operating phase for Qawali learner")
