from . import utils
from . import optim
from . import distributed
from . import load_dataset

from .model import get_model,DistEmbedLayer
from .gpu_cache import GPUCache
from .reduction import Feat,write_reductioned_feat,load_reductioned_feat,generate_reductioned_feat,write_mag_reductioned_feat
from .utils import extract_key_word_from_file,extract_str_from_file,draw_color,color_list