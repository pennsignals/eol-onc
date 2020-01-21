# eol-onc
### Machine Learning Approaches to Predict Six-Month Mortality among Patients with Cancer

Code associated with the manuscript by Ravi B. Parikh, MD, MPP, Christopher Manz, MD, Corey Chivers, PhD, Susan H. Regli, PhD, Jennifer Braun, MHA, Michael Draugelis, Lynn M. Schuchter, MD, Lawrence N. Shulman, MD, Mitesh S. Patel, MD, MBA, Nina O’Connor, MD.

Copyright (c) 2019 University of Pennsylvania Health System, MIT License

### Citation

```
@article{parikh2019machine,
  title={Machine Learning Approaches to Predict 6-Month Mortality Among Patients With Cancer},
  author={Parikh, Ravi B and Manz, Christopher and Chivers, Corey and Regli, Susan Harkness and Braun, Jennifer and Draugelis, Michael E and Schuchter, Lynn M and Shulman, Lawrence N and Navathe, Amol S and Patel, Mitesh S and others},
  journal={JAMA network open},
  volume={2},
  number={10},
  pages={e1915997--e1915997},
  year={2019},
  publisher={American Medical Association}
}
```

### Build & Run

Replace `/data/eol/eol-onc/` with the path to this project and run the following:

```bash
docker build -t eol .
docker run -d --rm -it -v /data/eol/eol-onc/:/data eol /bin/bash
```

### Training

Connect to the running container by finding the container name with `docker ps`

```bash
docker exec -it <container name> /bin/bash
```

In the container, spin up training jobs with:
```bash
cd /data
source activate eol_paper
nohup python3 EoL_model_ONC_v1_2.py -f data/OutpatientONC_v1_1_enc_data_features.csv --n-iter 100 --k-cv 5 -m rf > rf_gs.out &
disown %1

nohup python3 EoL_model_ONC_v1_2.py -f data/OutpatientONC_v1_1_enc_data_features.csv --n-iter 100 --k-cv 5 -m gb > gb_gs.out &
disown %1
```

