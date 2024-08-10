import os
from .chexbert import CheXbert
import numpy as np

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

CONDITIONS = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]

class CheXbertMetrics():
    def __init__(self, checkpoint_path, mbatch_size, device):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.chexbert = CheXbert(self.checkpoint_path, self.device,).to(self.device)

    def mini_batch(self, gts, res, mbatch_size=16):
        length = len(gts)
        assert length == len(res)
        for i in range(0, length, mbatch_size):
            yield gts[i:min(i + mbatch_size, length)], res[i:min(i + mbatch_size, length)]

    # def mini_batch(self, gts, res, mbatch_size=4):
    #     gts_items = list(gts.items())
    #     res_items = list(res.items())
    #     length = len(gts_items)
    #     for i in range(0, length, mbatch_size):
    #         gts_batch = dict(gts_items[i:min(i + mbatch_size, length)])
    #         res_batch = dict(res_items[i:min(i + mbatch_size, length)])
    #         yield gts_batch, res_batch

    def compute(self, gts, res):
        gts_chexbert = []
        res_chexbert = []
        for gt, re in self.mini_batch(gts, res, self.mbatch_size):
            gt_chexbert = self.chexbert(list(gt)).tolist()
            re_chexbert = self.chexbert(list(re)).tolist()
            gts_chexbert += gt_chexbert
            res_chexbert += re_chexbert
        gts_chexbert = np.array(gts_chexbert)
        res_chexbert = np.array(res_chexbert)

        res_chexbert = (res_chexbert == 1)
        gts_chexbert = (gts_chexbert == 1)

        tp = (res_chexbert * gts_chexbert).astype(float)
        tn = (~res_chexbert * ~gts_chexbert).astype(float)

        fp = (res_chexbert * ~gts_chexbert).astype(float)
        fn = (~res_chexbert * gts_chexbert).astype(float)

        tp_cls = tp.sum(0)
        fp_cls = fp.sum(0)
        fn_cls = fn.sum(0)
        tn_cls = tn.sum(0)

        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)
        tn_eg = tn.sum(1)

        precision_class = np.nan_to_num(tp_cls / (tp_cls + fp_cls))
        recall_class = np.nan_to_num(tp_cls / (tp_cls + fn_cls))
        f1_class = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls)))
        accuracy = np.nan_to_num((tp_eg + tn_eg) / (tp_eg + tn_eg + fp_eg + fn_eg)).mean()

        scores = {
            # example-based CE metrics
            'ce_precision': np.nan_to_num(tp_eg / (tp_eg + fp_eg)).mean(),
            'ce_recall': np.nan_to_num(tp_eg / (tp_eg + fn_eg)).mean(),
            'ce_f1': np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).mean(),
            'ce_accuracy': accuracy,  # 添加准确率到scores字典
            'ce_num_examples': float(len(res_chexbert)),
        }
        return scores

