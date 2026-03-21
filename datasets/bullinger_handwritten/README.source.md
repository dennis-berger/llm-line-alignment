# Bullinger Letters Dataset: CTC-Based Transcription Alignment

This dataset is part of a project aimed at improving annotation quality in historical handwritten documents using CTC-based alignment. It includes a corrected subset of 100 pages from the 16th-century Bullinger correspondence, aligned using dynamic programming and model output probabilities. The data supports research in handwritten text recognition, layout analysis, and iterative self-training approaches. The alignment algorithm is available via GitHub (https://github.com/andreas-fischer-unifr/nntp).


## Directory Structure

```
├── subset/
│ └── letter_id/
│ ├── page/
│ │ ├── 0001.xml
│ │ └── 0002.xml
│ ├── 0001.png
│ └── 0002.png
```

## Description

- **subset1, subset2, ...**: Top-level groupings of letter collections.
- **letter_id/**: Folder for each individual letter.
- **page/**: Contains XML files for each page of the letter.
  - `0001.xml`, `0002.xml`, ... — Page metadata or transcriptions.
- **.png files (outside `page/`)**: Scanned images corresponding to each page.
  - `0001.png`, `0002.png`, ... — Images of the original letter pages.


## Cite 

If you use this dataset in your work, please cite it as:

```bibtex
@inproceedings{peer2025bullinger,
  author    = {Marco Peer and Anna Scius-Bertrand and Andreas Fischer},
  title     = {{CTC Transcription Alignment of the Bullinger Letters: Automatic Improvement of Annotation Quality}},
  booktitle = {Proceedings of the 2nd International Workshop on Computer Vision Systems for Document Analysis and Recognition (VisionDocs)},
  year      = {2025}
}