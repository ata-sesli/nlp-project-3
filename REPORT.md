\# Uzbek Speech Recognition: Experimental Report



\## 1. Introduction



This project investigates automatic speech recognition (ASR) for Uzbek using the Latin script. We evaluate conventional and neural ASR systems on the Uzbek Speech Corpus (USC), comparing zero-shot pretrained models, models trained from scratch, and models fine-tuned on the project training split.



All systems are evaluated using a shared test set and a common normalization and scoring pipeline. The main evaluation metrics are Word Error Rate (WER) and Character Error Rate (CER). We also examine inference efficiency through the Real-Time Factor (RTF).



\## 2. Dataset



The experiments use the Uzbek Speech Corpus provided by ISSAI.



The project constructs speaker-disjoint training, development, and test splits. Development and test speakers are excluded from the training selection to prevent speaker overlap.



| Split | Clips | Speakers | Audio duration |

|---|---:|---:|---:|

| Train | 27,144 | Speaker-disjoint | 23.92 hours |

| Development | 3,783 | 41 | 3.95 hours |

| Test | 3,836 | 38 | 4.40 hours |



The training split was selected under a 25-hour audio budget. After cleaning, 23.92 hours were retained. The development and test splits were kept complete, apart from the documented cleaning and correction procedures.



Data preparation includes audio validation, transcript normalization, duplicate-audio handling, and checks for invalid or inconsistent text. The detailed cleaning procedures and dataset licensing information are documented in `data/README.md` and the root `README.md`.



\## 3. Experimental Systems



The evaluated systems are divided into two main groups.



\### 3.1. Baseline systems



\- A1: GMM-HMM trained using Kaldi.

\- A2: Conformer-CTC trained from scratch.

\- A3: Conformer-RNNT trained from scratch.

\- A4: Whisper-small, zero-shot.

\- A5: Whisper-large-v3, zero-shot.
- A6: Meta MMS-1B with the Uzbek Latin language adapter, zero-shot on USC.
- A7: XLS-R Uzbek Common Voice 8, zero-shot on USC.

\- A8: NVIDIA Uzbek FastConformer, zero-shot RNNT.

\- A9: NVIDIA Uzbek FastConformer, zero-shot CTC.



\### 3.2. Fine-tuned systems



\- C1: Whisper-small fine-tuned on the project training data.

\- C2: XLS-R 300M with CTC fine-tuning.

\- C2long: Extended training of XLS-R 300M with CTC.

\- C4a: NVIDIA FastConformer fine-tuned using the RNNT head.

\- C4b: NVIDIA FastConformer fine-tuned using the CTC head.



The C4 experiments start from the pretrained NVIDIA Uzbek FastConformer hybrid model. The model contains approximately 114.6 million parameters. Fine-tuning uses the project training split and preserves the model's original SentencePiece tokenizer.



\## 4. Evaluation Methodology



All systems are evaluated on the same 3,836 test clips.



The benchmark uses a shared normalization pipeline and computes normalized WER and CER. This ensures that model comparisons use consistent text-processing rules.



The Real-Time Factor is used to measure inference efficiency:



\\\[

RTF = \\frac{\\text{Inference time}}{\\text{Audio duration}}

\\]



An RTF below 1 indicates that inference is faster than real time.



For statistical uncertainty, speaker-level bootstrap confidence intervals are calculated using 10,000 resamples. Paired comparisons between selected fine-tuned systems and their corresponding pretrained baselines are evaluated using p-values with Holm correction for multiple comparisons.



\## 5. Test Results



\### 5.1. Baseline performance



| System | WER | CER |

|---|---:|---:|

| A1 GMM-HMM | 60.1028% | 28.7577% |

| A2 Conformer-CTC | 50.1738% | 13.9743% |

| A3 Conformer-RNNT | 42.1862% | 16.2412% |

| A4 Whisper-small | 119.8839% | 53.1816% |

| A5 Whisper-large-v3 | 93.6580% | 32.6871% |
| A6 MMS-1B Uzbek | 29.3802% | 6.6048% |
| A7 XLS-R Uzbek CV8 | 40.7144% | 8.7757% |

| A8 NVIDIA FastConformer RNNT | 12.0664% | 3.6533% |

| A9 NVIDIA FastConformer CTC | 13.0834% | 3.6124% |



Among the evaluated baseline systems, NVIDIA FastConformer RNNT achieves the lowest WER, while NVIDIA FastConformer CTC achieves the lowest CER.



The Whisper zero-shot systems perform substantially worse on this test set than the Uzbek-oriented FastConformer models. Among the newly evaluated systems, MMS-1B achieves lower WER and CER than XLS-R Uzbek CV8, while XLS-R provides faster inference.



\### 5.2. Fine-tuned performance



| System | WER | CER |

|---|---:|---:|

| C1 Whisper-small fine-tuned | 35.5262% | 9.8929% |

| C2 XLS-R 300M CTC | 35.4042% | 8.4513% |

| C2long XLS-R 300M extended | 32.9598% | 7.7840% |

| C4a FastConformer RNNT fine-tuned | 11.3823% | 3.4067% |

| C4b FastConformer CTC fine-tuned | 12.2476% | 3.3132% |



C4a achieves the lowest WER among all evaluated systems. C4b achieves the lowest CER.



Fine-tuning the NVIDIA FastConformer improves both WER and CER relative to its corresponding zero-shot configuration. The XLS-R extended training experiment also improves over the shorter XLS-R fine-tuning run.



\## 6. Inference Efficiency



The recorded inference results show that the neural systems can process the test audio substantially faster than real time.



| System | RTF |

|---|---:|

| A1 GMM-HMM | 0.21749 |

| A2 Conformer-CTC | 0.00062 |

| A3 Conformer-RNNT | 0.00147 |

| A4 Whisper-small | 0.03790 |

| A5 Whisper-large-v3 | 0.08499 |

| A6 MMS-1B Uzbek | 0.01178 |
| A7 XLS-R Uzbek CV8 | 0.00546 |

| A8 NVIDIA FastConformer RNNT | 0.00488 |

| A9 NVIDIA FastConformer CTC | 0.00319 |

| C1 Whisper-small fine-tuned | 0.02012 |

| C2 XLS-R 300M CTC | 0.00384 |

| C2long XLS-R extended | 0.00402 |

| C4a FastConformer RNNT fine-tuned | 0.00433 |

| C4b FastConformer CTC fine-tuned | 0.00321 |



The reported GPU RTF measurements were obtained on a Tesla T4. A1 was measured on CPU hardware, so its RTF should not be interpreted as a direct hardware-controlled comparison with the GPU systems.



\## 7. Statistical Analysis



Speaker-level bootstrap confidence intervals were computed using 10,000 resamples.



For C4a compared with A8:



\- WER difference: -0.006841.

\- 95% confidence interval: \[-0.021013, 0.002505].

\- Raw p-value: 0.191781.

\- Holm-adjusted p-value: 0.237176.



For C4b compared with A9:



\- WER difference: -0.008357.

\- 95% confidence interval: \[-0.027040, 0.001403].

\- Raw p-value: 0.118588.

\- Holm-adjusted p-value: 0.237176.



Neither WER comparison is statistically significant after Holm correction. The confidence intervals include zero, so the observed improvements should not be interpreted as conclusive evidence of a population-level improvement.



The XLS-R extended experiment shows statistically significant improvements over the shorter XLS-R experiment:



\- WER difference: -0.024443; Holm-adjusted p-value: 0.000800.

\- CER difference: -0.006674; Holm-adjusted p-value: 0.000800.



The Whisper-small fine-tuning comparison against its zero-shot baseline is also statistically significant in both WER and CER under the paired test.



\## 8. Discussion



The experiments demonstrate that Uzbek-specific pretrained acoustic models provide a strong starting point for Uzbek ASR.



The NVIDIA FastConformer systems substantially outperform the other evaluated zero-shot baselines. Fine-tuning the same architecture on the project training data produces the best observed WER and CER results across the evaluated systems.



The newly evaluated MMS-1B baseline achieves 29.38% WER and 6.60% CER, outperforming the XLS-R Uzbek CV8 baseline (40.71% WER and 8.78% CER). XLS-R is faster, with an RTF of 0.00546 compared with MMS-1B's 0.01178. Both systems are substantially less accurate than the NVIDIA FastConformer baselines.

The distinction between WER and CER is important. C4a obtains the lowest WER, whereas C4b obtains the lowest CER. This indicates that the preferred model depends on whether word-level or character-level transcription errors are prioritized.



The statistical analysis also highlights the importance of uncertainty estimates. Although C4a has a numerically lower WER than A8, the paired comparison does not establish a statistically significant improvement. The result should therefore be reported as an observed numerical gain rather than a confirmed significant effect.



\## 9. Reproducibility



The repository includes the data preparation pipeline, model-specific experiment directories, prediction files, benchmark scripts, result metadata, and automated tests.



Relevant components include:



\- `data/`: dataset preparation, cleaning, normalization, and statistics.

\- `baselines/`: baseline ASR systems.

\- `whisper\_finetune/`: Whisper fine-tuning.

\- `xlsr\_ctc/`: XLS-R fine-tuning.

\- `nvidia\_finetune/`: NVIDIA FastConformer fine-tuning.

\- `benchmark/`: evaluation, timing, statistical analysis, and predictions.

\- `tests/`: automated tests for core project components.



The benchmark and statistical analysis can be reproduced using the scripts and environment requirements documented in the repository.



\## 10. Conclusion



This project evaluates multiple ASR architectures for Uzbek using a shared dataset and benchmark pipeline.



NVIDIA FastConformer fine-tuned with the RNNT head achieves the best observed WER of 11.3823%, while the CTC fine-tuned variant achieves the best observed CER of 3.3132%.



The results support the effectiveness of Uzbek-oriented pretrained acoustic models and task-specific fine-tuning. However, the paired statistical comparisons between the fine-tuned FastConformer models and their zero-shot counterparts do not show statistically significant WER improvements after multiple-comparison correction.



Future work could investigate larger training datasets, alternative decoding strategies, language-model integration, and more extensive evaluation across additional Uzbek speech domains.




