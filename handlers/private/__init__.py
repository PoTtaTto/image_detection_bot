# Third-party
from aiogram import Router

# Routers
from .basic import basic_router
from .image import image_router

private_router = Router()
sub_routers = [
    basic_router, image_router
]

private_router.include_routers(*sub_routers)
