import os
from pathlib import Path

from dotenv import load_dotenv
from dependency_injector import containers, providers

from src.trainer.trainer import Trainer


load_dotenv()


class TrainerContainer(containers.DeclarativeContainer):    
    trainer = providers.Singleton(
        Trainer,
    )

    default_trainer = providers.Singleton(
        Trainer,
    )