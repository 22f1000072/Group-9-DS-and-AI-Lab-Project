# Milestone 1: Problem and Literature

## Problem definition
Noise-Robust Speech Recognition and Error Detection
An Intelligent Audio Transcription and Verification Assistant
Automatic Speech Recognition (ASR) loses accuracy on noisy recordings such as street interviews, markets, vehicles and field surveys. Modern recognizers also produce fluent text even when they are wrong, so errors are silent: a wrong number or name looks as confident as a correct one. Users therefore re-listen to the whole recording to verify it, which removes most of the time ASR was meant to save. This project studies how noise type and level affect a pretrained recognizer, trains a noise-robust version, and builds, a transcription assistant that flags the words most likely to be wrong, links each flag to its exact position in the original audio, and lets users replay and correct only those parts.


## Proposed direction

TODO

## References
[1] Graves, A., Fernández, S., Gomez, F., Schmidhuber, J. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. ICML, 2006.
[2] Baevski, A., Zhou, H., Mohamed, A., Auli, M. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS, 2020. arXiv:2006.11477.
[3] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I. Robust Speech Recognition via Large-Scale Weak Supervision. ICML, 2023. arXiv:2212.04356.
[4] Panayotov, V., Chen, G., Povey, D., Khudanpur, S. Librispeech: An ASR Corpus Based on Public Domain Audio Books. ICASSP, 2015.
[5] Likhomanenko, T., Xu, Q., Pratap, V., Tomasello, P., Kahn, J., Avidov, G., Collobert, R., Synnaeve, G. Rethinking Evaluation in ASR: Are Our Models Robust Enough? Interspeech, 2021. arXiv:2010.11745.
[6] Iwamoto, K., Ochiai, T., Delcroix, M., Ikeshita, R., Sato, H., Araki, S., Katagiri, S. How Bad Are Artifacts?: Analyzing the Impact of Speech Enhancement Errors on ASR. Interspeech, 2022. arXiv:2201.06685.
[7] Koenecke, A., Choi, A. S. G., Mei, K., Schellmann, H., Sloane, M. Careless Whisper: Speech-to-Text Hallucination Harms. ACM FAccT, 2024. DOI: 10.1145/3630106.3658996.
[8] Piczak, K. J. ESC: Dataset for Environmental Sound Classification. ACM Multimedia, 2015.
[9] Wolf, T. et al. Transformers: State-of-the-Art Natural Language Processing. EMNLP (System Demonstrations), 2020.
[10] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On Calibration of Modern Neural Networks. ICML, 2017. arXiv:1706.04599.
[11] Jiang, H. Confidence Measures for Speech Recognition: A Survey. Speech Communication, 45(4), 2005.
[12] Watanabe, S. et al. CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings. CHiME Workshop, 2020. arXiv:2004.09249.
[13] Schröter, H., Escalante-B., A. N., Rosenkranz, T., Maier, A. DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio Based on Deep Filtering. ICASSP, 2022. arXiv:2110.05588.
[14] Défossez, A., Synnaeve, G., Adi, Y. Real Time Speech Enhancement in the Waveform Domain. Interspeech, 2020. arXiv:2006.12847.
Tools used: Hugging Face Transformers and Datasets, evaluate and JiWER, librosa, PyTorch, Gradio, Kaggle (T4 GPU).

