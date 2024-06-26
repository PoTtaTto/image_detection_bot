# 🤖 Telegram Object Detection Bot

Этот репозиторий содержит Telegram чат-бота, написанного на Python 3.11.0, который позволяет распознавать объекты на изображении. Для этого используются библиотеки `aiogram` и `opencv`, а также готовая модель `ssd_mobilnet_v3_large_coco`.
Ссылка чат-бота: https://t.me/ImageDetection_MobileNetBot

## 📝 Пример работы

После запуска бота, вы можете взаимодействовать с ним в Telegram:

- `/start` - приветственное сообщение от бота
- `/help` - список доступных команд
- Отправка изображения - получение в ответ тоже самое изображение с обведенными объектами и их названием


---


![Пример работы 1](https://github.com/PoTtaTto/image_detection_bot/assets/69969678/9178a9f1-b379-4e9b-b1de-5c0974531ef5)

**Базовые команды /start и /help**

---


![Пример работы 2](https://github.com/PoTtaTto/image_detection_bot/assets/69969678/6ae3f948-6014-449b-87ed-8ca4b8dcafaa)

**Отправка изображения**

## 🚀 Установка

### 🔧 Через виртуальное окружение

Для начала убедитесь, что на вашем компьютере предустановлен `Python 3.11.0` или выше.

1. Клонируйте репозиторий:
    
```bash
    git clone https://github.com/PoTtaTto/image_detection_bot
```


2. Перейдите в директорию проекта:
    
```bash
    cd image_detection_bot
```    


3. Создайте виртуальное окружение:
    
```bash
    python -m venv .venv
```


4. Активируйте виртуальное окружение:
- Windows:
        
```bash
        .venv\Scripts\activate
```


- macOS и Linux:

        
```bash
        source .venv/bin/activate
```


5. Установите зависимости из `requirements.txt`:
    
```bash
    pip install -r requirements.txt
```


6. Запустите бота:
    
```bash
    python start.py
```


### 🐳 Через Docker

Для этого метода убедитесь, что на вашем компьютере предустановлен `Docker`.

1. Клонируйте репозиторий:
    
```bash
    git clone https://github.com/PoTtaTto/image_detection_bot
```


2. Перейдите в директорию проекта:
    
```bash
    cd image_detection_bot
```


3. Постройте и запустите контейнер:
    
```bash
    docker compose up --build
```


   🛑 **Примечание:** В связи с блокировкой Docker на территории РФ 30.05.2024 года, вам может потребоваться изменить конфигурационный файл Docker. Подробнее [тут](https://proglib.io/p/docker-ushel-iz-rf-instrukciya-po-vosstanovleniyu-dostupa-k-docker-hub-dlya-polzovateley-iz-rossii-2024-05-30). Данный метод не является безопасным и применяется на свой страх и риск.

## 📌 Замечания

- Убедитесь, что токен чат-бота сохранен в файле .env в переменной BOT_TOKEN.
