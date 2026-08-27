"""
Models Module
=============
"""

from .deep_models import (
    CNN1DEncoder,
    LSTMEncoder,
    TransformerEncoder,
    CNN_LSTMEncoder,
    SingleTaskModel,
    MultiTaskModel,
    create_model
)

from .hierarchical_mtl import (
    HierarchicalMTLModel,
    HierarchyConstraint,
    UncertaintyWeighting as HierarchicalUncertaintyWeighting,
    GradientSurgery
)

from .mtl_models import (
    HardSharingMTL,
    SoftSharingMTL,
    CrossStitchMTL,
    MMoEMTL,
    PLEMTL,
    MTANMTL,
    create_mtl_model,
    MTL_MODEL_REGISTRY,
    DEFAULT_TASKS
)

from .gradient_methods import (
    GradientMethod,
    NoneGradient,
    PCGrad,
    GradNorm,
    CAGrad,
    MGDA,
    UncertaintyWeighting,
    DWA,
    create_gradient_method,
    GRADIENT_METHOD_REGISTRY
)

from .losses import (
    FocalLoss,
    ClassBalancedLoss,
    HierarchyConsistencyLoss,
    MixupLoss,
    MTLLoss
)

__all__ = [
    # Encoders
    'CNN1DEncoder',
    'LSTMEncoder', 
    'TransformerEncoder',
    'CNN_LSTMEncoder',
    
    # Single/Multi Task Models
    'SingleTaskModel',
    'MultiTaskModel',
    'HierarchicalMTLModel',
    
    # MTL Architectures
    'HardSharingMTL',
    'SoftSharingMTL',
    'CrossStitchMTL',
    'MMoEMTL',
    'PLEMTL',
    'MTANMTL',
    'create_mtl_model',
    'MTL_MODEL_REGISTRY',
    'DEFAULT_TASKS',
    
    # Gradient Methods
    'GradientMethod',
    'NoneGradient',
    'PCGrad',
    'GradNorm',
    'CAGrad',
    'MGDA',
    'UncertaintyWeighting',
    'DWA',
    'create_gradient_method',
    'GRADIENT_METHOD_REGISTRY',
    
    # Loss Functions
    'FocalLoss',
    'ClassBalancedLoss',
    'HierarchyConsistencyLoss',
    'MixupLoss',
    'MTLLoss',
    
    # Components (legacy)
    'HierarchyConstraint',
    'HierarchicalUncertaintyWeighting',
    'GradientSurgery',
    
    # Factory
    'create_model'
]
