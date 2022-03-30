# Framework
This implementation is based on the framework for CTR predictions found @ https://github.com/shenweichen/DeepCTR-Torch.<br/>
(Recommended) Requirements are: Python (3.9.12), tensorflow (2.8.0), sklearn, xgboost (1.5.2), torch (1.11.0), tqdm and pandas. (Note it might well work with other versions but it was not tested) 
# FMMixer
Brief: FM trained together with a MLPMixer for CTR prediction.<br/><br/>
![Full Model](https://i.imgur.com/F4rH49g.png)
*Figure 1: Full FMMixer model* <br/><br/><br/><br/><br/>
![MLP Mixer](https://i.imgur.com/5NUfFzK.png)
*Figure 2: Mixer component* <br/><br/><br/><br/><br/>
![Mixer Layer](https://i.imgur.com/QAy9Jmb.png)
*Figure 3: Single mixer layer* <br/><br/><br/><br/><br/>
# How to run
Sample data (100K entries) is already present, to do a full run with preprocessing and training run [Avazu version](Avazu/fmmixer_avazu.py) and [Criteo version](Criteo/fmmixer_criteo.py). <br/> Note: you might need to add the project folder to PYTHONPATH, or sys.path.append() depending on your user case.
