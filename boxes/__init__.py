#from .callbacks import *
#from .probability_dataset import *
#from .loss_functions import *
#from .box_operations import *
#from .metrics import *
#from .modules import *
#
## This is to handle this bug in tqdm, which is fixed in Jupyter but not in JupyterLab:
##   https://github.com/tqdm/tqdm/issues/433
## Workaround:
##   https://github.com/bstriner/keras-tqdm/issues/21
#try:
#    from IPython import get_ipython
#    if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
#        raise ImportError("console")
#except:
#    pass
#else:
#    from IPython.core.display import HTML, display
#    display(
#        HTML("""
#    <style>
#    .p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {
#      padding: 0;
#      border: 0;
#    }
#    </style>
#    """))
