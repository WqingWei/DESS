"""
@article{paparrizos2022volume,
  title={{Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection}},
  author={Paparrizos, John and Boniol, Paul and Palpanas, Themis and Tsay, Ruey S and Elmore, Aaron and Franklin, Michael J},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={11},
  pages={2774--2787},
  year={2022},
  publisher={VLDB Endowment}
}

""" 

from typing import List
import numpy as np

def get_list_anomaly(labels: np.ndarray) -> List[int]:
   
    # results = []
    # start = 0
    # anom = False
    # for i, val in enumerate(labels):
    #     if val == 1:
    #         anom = True
    #     else:
    #         if anom:
    #             results.append(i - start)
    #             anom = False
    #     if not anom:
    #         start = i
    # return results

    end_pos = np.diff(np.array(labels, dtype=int), append=0) < 0
    return np.diff(np.cumsum(labels)[end_pos], prepend=0)
