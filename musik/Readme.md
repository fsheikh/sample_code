Workspace for MIR experiments with eastern classical/semi-classical
music. At the time of writing focus is solely on Qawali songs.

This directory contains:

* Reference qawali data-set builder
    * Includes metadata describing data-set in json
    * qawali_dataset_builder: Given above json, prepares data-set
    either from already downloaded songs or downloads them on demand

* Qawali genre detector
    * taali_separator: Unsupervised detection of Qawali genre

* Miscellaneous:
    * Utility scripts to run genre detection algorithms on raw audio

* Dependencies
    * librosa
    * NUSSL
    * lmfit