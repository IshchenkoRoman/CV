import os
from pathlib import Path

from dotenv import load_dotenv
from dependency_injector import containers, providers


from src.metrics.losses.bce_loss import BCELoss
from src.metrics.losses.dice_loss import DiceLoss
from src.metrics.losses.focal_loss import FocalLoss
from src.metrics.quality.iou import IOU


load_dotenv()


class MetricsContainer(containers.DeclarativeContainer):
    """  
    A container class for managing metric and loss function instances.  

    This class uses dependency injection to create singleton instances of various 
    loss functions and evaluation metrics, ensuring a single instance is used 
    throughout the application.

    """ 
    
    bce_loss = providers.Singleton(
        BCELoss,
    )

    dice_loss = providers.Singleton(
        DiceLoss,
    )

    focal_loss = providers.Singleton(
        FocalLoss,
    )

    iou_metric = providers.Singleton(
        IOU,
    )

    default_loss = providers.Singleton(
        DiceLoss,
    )