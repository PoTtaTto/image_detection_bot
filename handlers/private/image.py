# Third-party
from aiogram import Router, F
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

# Standard
from pathlib import Path
import aiohttp

# Project
import config as cf
from logger import bot_logger
from detection import detect_and_draw_boxes


# __router__ !DO NOT DELETE!
image_router = Router()


# __states__ !DO NOT DELETE!


# __buttons__ !DO NOT DELETE!


# __chat__ !DO NOT DELETE!
async def __download_image(url: str, destination: str | Path) -> bool:
    """Download image from URL to destination folder.

    Args:
        url (str): URL of the image.
        destination (str | Path): Path where the image will be saved.

    Returns:
        bool: True if image download was successful, False otherwise.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(destination, "wb") as file:
                    file.write(content)
                bot_logger.warning(f"Image downloaded successfully to {destination}")
                return True
            else:
                bot_logger.error(f"Failed to download image. Status code: {response.status}")
                return False


@image_router.message(F.photo)
async def handle_photo_command(message: Message, state: FSMContext):
    """Handle photo command from user.

    Args:
        message (Message): Input message with photo.
        state (FSMContext): State of the finite state machine.
    """
    bot_logger.info(f'Handling image from user {message.chat.id}')

    file_id = message.photo[-1].file_id
    file = await message.bot.get_file(file_id=file_id)
    url = f'https://api.telegram.org/file/bot{cf.bot["token"]}/{file.file_path}'

    download_path = cf.DATA_PATH / 'images' / f'{file_id}.jpg'
    await __download_image(url=url, destination=download_path)

    result_image_path = detect_and_draw_boxes(image_path=download_path, scale_factor=1, name=f'result_{file_id}.jpg')

    from aiogram.types import FSInputFile
    output_image = FSInputFile(path=result_image_path)

    await message.bot.send_photo(
        chat_id=message.chat.id,
        photo=output_image
    )