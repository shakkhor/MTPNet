# MTPNet
Code of paper [Multi-scale Transformer Pyramid Networks for Multivariate Time Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10412052)

MTPNet achieves SOTA on nine benchmarks.

## Introduction
Multivariate Time Series (MTS) forecasting involves modeling temporal dependencies within historical records. Transformers have demonstrated remarkable performance in MTS forecasting due to their capability to capture long-term dependencies. However, prior work has been confined to modeling temporal dependencies at either a fixed scale or multiple scales that exponentially increase (most with base 2). This limitation hinders their effectiveness in capturing diverse seasonalities, such as hourly and daily patterns. In this paper, we introduce a dimension invariant embedding technique that captures short-term temporal dependencies and projects MTS data into a higher-dimensional space, while preserving the dimensions of time steps and variables in MTS data. Furthermore, we present a novel Multi-scale Transformer Pyramid Network (MTPNet), specifically designed to effectively capture temporal dependencies at multiple unconstrained scales. The predictions are inferred from multi-scale latent representations obtained from transformers at various scales. Extensive experiments on nine benchmark datasets demonstrate that the proposed MTPNet outperforms recent state-of-the-art methods.

## Train and Test
1. Install the required packages: `pip install -r requirements.txt`
2. Data are publicly available at [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1).
3. To reproduce the experimental results presented in the paper. Simply run the scripts at "/MTPNet/scripts/" as follows:
   ```
   bash ./scripts/ETTh1.sh
   bash ./scripts/ETTh2.sh
   bash ./scripts/ETTm1.sh
   bash ./scripts/ETTm2.sh
   bash ./scripts/electricity.sh
   bash ./scripts/Exchange_Rate.sh
   bash ./scripts/Traffic.sh
   bash ./scripts/WTH.sh
   bash ./scripts/ILI.sh
   ```

## Run visualizer
Every test run writes `run_info.json` and `report.json` into its `results/<run>/` folder. To browse them, run from the repo root:
```
python visualizer/serve.py
```
It opens a page in your browser listing every run with its parameters and metrics. Tick up to six runs on the same dataset to see their predictions together against the actual values, by day, week, month or year, split into trend, seasonal and local parts, with an hour-by-hour table you can filter by how far off a prediction was. Options: `--results <folder>`, `--port <n>`, `--host <address>`, `--no-browser`.

On a remote machine (SSH or a remote IDE session) it does not try to open a browser. It prints the port instead: let your editor forward it (VS Code's Ports panel, PyCharm/Gateway's forwarded ports), or run the `ssh -N -L ...` command it prints on your own computer, then open `http://127.0.0.1:8765/` locally.

## Citation
If you find this repository beneficial for your research, kindly include a citation:
```
@ARTICLE{10412052,
  author={Zhang, Yifan and Wu, Rui and Dascalu, Sergiu M. and Harris, Frederick C.},
  journal={IEEE Access}, 
  title={Multi-Scale Transformer Pyramid Networks for Multivariate Time Series Forecasting}, 
  year={2024},
  volume={12},
  number={},
  pages={14731-14741},
  keywords={Transformers;Forecasting;Decoding;Market research;Time series analysis;Predictive models;Data models;Time series analysis;Time series forecasting;transformer;multi-scale feature pyramid;value embedding},
  doi={10.1109/ACCESS.2024.3357693}}
```

## Acknowledgements
We sincerely appreciate the foundational code from the following GitHub repositories: \
https://github.com/wanghq21/MICN \
https://github.com/zhouhaoyi/Informer2020 \
https://github.com/Thinklab-SJTU/Crossformer \
https://github.com/thuml/Time-Series-Library \
https://github.com/cure-lab/SCINet
