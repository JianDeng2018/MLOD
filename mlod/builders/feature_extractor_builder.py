from mlod.core.feature_extractors.bev_vgg import BevVgg
from mlod.core.feature_extractors.bev_resnet import BevResnet
from mlod.core.feature_extractors.bev_inception import BevInception
from mlod.core.feature_extractors.bev_vgg_pyramid import BevVggPyr
from mlod.core.feature_extractors.bev_vgg_lfe import BevVggLfe
from mlod.core.feature_extractors.bev_vgg_concat import BevVggConcat
from mlod.core.feature_extractors.depth_vgg import DepthVgg
from mlod.core.feature_extractors.img_vgg import ImgVgg
from mlod.core.feature_extractors.img_resnet import ImgResnet
from mlod.core.feature_extractors.img_inception import ImgInception
from mlod.core.feature_extractors.img_vgg_pyramid import ImgVggPyr
from mlod.core.feature_extractors.img_vgg_pyramid2 import ImgVggPyr2
from mlod.core.feature_extractors.img_vgg_pyramid3 import ImgVggPyr3
from mlod.core.feature_extractors.img_vgg_retina import ImgVggRtn
from mlod.core.feature_extractors.img_vgg_concat import ImgVggConcat
from mlod.core.feature_extractors.img_vgg_pure import ImgVggPure


def get_extractor(extractor_config):

    extractor_type = extractor_config.WhichOneof('feature_extractor')

    # BEV feature extractors
    if extractor_type == 'bev_vgg':
        return BevVgg(extractor_config.bev_vgg)
    elif extractor_type == 'bev_vgg_pyr':
        return BevVggPyr(extractor_config.bev_vgg_pyr)
    elif extractor_type == 'bev_vgg_concat':
        return BevVggConcat(extractor_config.bev_vgg_concat)
    elif extractor_type == 'bev_resnet':
        return BevResnet(extractor_config.bev_resnet)
    elif extractor_type == 'bev_inception':
        return BevInception(extractor_config.bev_inception)
    elif extractor_type == 'bev_vgg_lfe':
        return BevVggLfe(extractor_config.bev_vgg_lfe)

    # Image feature extractors
    elif extractor_type == 'img_vgg':
        return ImgVgg(extractor_config.img_vgg)
    elif extractor_type == 'img_vgg_pyr':
        return ImgVggPyr(extractor_config.img_vgg_pyr)
    elif extractor_type == 'img_vgg_pyr2':
        return ImgVggPyr2(extractor_config.img_vgg_pyr2)
    elif extractor_type == 'img_vgg_pyr3':
        return ImgVggPyr3(extractor_config.img_vgg_pyr3)
    elif extractor_type == 'img_vgg_retina':
        return ImgVggRtn(extractor_config.img_vgg_retina)
    elif extractor_type == 'img_vgg_concat':
        return ImgVggConcat(extractor_config.img_vgg_concat)
    elif extractor_type == 'img_vgg_pure':
        return ImgVggPure(extractor_config.img_vgg_pure)
    elif extractor_type == 'depth_vgg':
        return DepthVgg(extractor_config.depth_vgg)
    elif extractor_type == 'img_resnet':
        return ImgResnet(extractor_config.img_resnet)
    elif extractor_type == 'img_inception':
        return ImgInception(extractor_config.img_inception)

    return ValueError('Invalid feature extractor type', extractor_type)
