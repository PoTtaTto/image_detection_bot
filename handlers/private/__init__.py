# Third-party
from aiogram import Router

# Routers
from .basic import basic_router

private_router = Router()
sub_routers = [
    basic_router
]

private_router.include_routers(*sub_routers)
