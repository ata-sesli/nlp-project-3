# Data: Uzbek Speech Corpus (USC)

All audio and transcripts in `data/raw/` come from the Uzbek Speech Corpus
(USC) by ISSAI and TUIT: 105 hours of read Uzbek speech from 958 speakers,
with transcripts checked by native speakers.

- Download: [issai/Uzbek_Speech_Corpus](https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus) on Hugging Face (`ISSAI_USC.tar.gz`, 9.47 GB)
- Paper: [USC: An Open-Source Uzbek Speech Corpus and Initial Speech Recognition Experiments](https://arxiv.org/abs/2107.14419)
- DOI: [10.48342/drss-8q87](https://doi.org/10.48342/drss-8q87)

## License

The two official sources state different licenses:

- the Hugging Face dataset card: **MIT**;
- [ISSAI's USC page](https://issai.nu.edu.kz/uzbek-asr/): **Creative Commons Attribution 4.0 International (CC BY 4.0)**, for academic and commercial use.

The repository's Apache-2.0 `LICENSE` covers our code only, not this data.

## Citation

```bibtex
@inproceedings{musaev2021usc,
  title={USC: An open-source Uzbek speech corpus and initial speech recognition experiments},
  author={Musaev, Muhammadjon and Mussakhojayeva, Saida and Khujayorov, Ilyos and Khassanov, Yerbolat and Ochilov, Mannon and Atakan Varol, Huseyin},
  booktitle={Speech and Computer: 23rd International Conference, SPECOM 2021},
  pages={437--447},
  year={2021},
  organization={Springer}
}
```

## What is stored here

Only the clips of our splits are stored, not the full corpus: our 25-hour train
selection and the complete official dev and test sets. The audio is converted
from the original WAV to lossless FLAC; the transcripts are the unchanged
original files. See the root `README.md` for how the splits are made and
`data/get_raw.py` for the conversion.
