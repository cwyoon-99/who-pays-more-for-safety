"""P-MMEval subsets used in the paper, each in its 10 languages.

Flores-200 and HumanEval-XL are excluded (Table capability_drop). The dataset configs
are the stock OpenCompass P-MMEval configs at the pinned commit.

The summarizer lists every per-language subset so that the summary CSV written by
OpenCompass contains the rows that ``capability/summarize.py`` needs.
"""

from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.PMMEval.mgsm_gen_679720 import PMMEval_MGSM_datasets
    from opencompass.configs.datasets.PMMEval.mhellaswag_gen_1a6b73 import PMMEval_MHellaswag_datasets
    from opencompass.configs.datasets.PMMEval.mifeval_gen_79f8fb import PMMEval_MIFEval_datasets
    from opencompass.configs.datasets.PMMEval.mlogiqa_gen_36c4f9 import PMMEval_MLogiQA_datasets
    from opencompass.configs.datasets.PMMEval.mmmlu_gen_d5017d import PMMEval_MMMLU_datasets
    from opencompass.configs.datasets.PMMEval.xnli_gen_973734 import PMMEval_XNLI_datasets
    from opencompass.configs.summarizers.groups.PMMEval import PMMEval_summary_groups

PMMEval_paper_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# mIFEval is scored with strict accuracy, every other subset with accuracy.
summarizer = dict(
    dataset_abbrs=[
        ('mhellaswag', 'accuracy'), ('mifeval', 'strict_acc'), ('mlogiqa', 'accuracy'),
        ('mmmlu', 'accuracy'), ('xnli', 'accuracy'), ('mgsm', 'accuracy'),
    ] + sum([[(subset, 'strict_acc' if group['name'] == 'mifeval' else 'accuracy')
              for subset in group['subsets']]
             for group in PMMEval_summary_groups
             if group['name'] in ('mhellaswag', 'mifeval', 'mlogiqa', 'mmmlu', 'xnli', 'mgsm')], []),
    summary_groups=PMMEval_summary_groups,
)
