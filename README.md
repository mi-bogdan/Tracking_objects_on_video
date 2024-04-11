<h2 align="center">ОТСЛЕЖИВАНИЕ ОБЪЕКТОВ НА ВИДЕО</h2>

**Ссылки**:
- [Вконтакте](https://vk.com/id404101172)
- [Telegram](https://t.me/bogdan_shnyra)

## Описание

#####  Модель:
    https://github.com/C-Aniruddh/realtime_object_recognition

Скачивать два файла:

- MobileNetSSD_deploy.caffemodel
- MobileNetSSD_deploy.prototxt.txt


Код для обнаружения объектов в реальном времени и отслеживания потока с помощью OpenCV. Программа использует предварительно обученную модель MobileNetSSD для идентификации объектов в видеопотоках и применяет метод Фарнбека для отслеживания оптического потока с целью визуализации динамики движения.
Используемая модель MobileNet SSD — это легкая глубокая нейронная сеть, которая обнаруживает объекты на видео. Файлы модели (`MobileNetSSD_deploy.prototxt.txt` и `MobileNetSSD_deploy.caffemodel`) должны быть размещены в том же каталоге, что и скрипт.

##### Особенности

- **Обнаружение объектов в реальном времени**: Обнаруживает несколько объектов из предопределенных классов в видеокадрах, получаемых с веб-камеры.
- **Отслеживание объектов**: После обнаружения объекты отслеживаются по кадрам, что помогает понять движение объекта.
- **Расчет и отображение FPS**: Оценка и отображение количества кадров в секунду.
- **Визуализация оптического потока**: Визуализирует движение между двумя последовательными кадрами для понимания динамики движений в сцене с помощью метода Фарнбека.

### Инструменты разработки

**Стек:**
- Python
- Numpy
- OpenCV

## Разработка

##### 1) Клонировать репозиторий

    git clone https://github.com/mi-bogdan/Tracking_objects_on_video.git

##### 2) Создать виртуальное окружение

    python -m venv venv

##### 3) Активировать виртуальное окружение

    venv/Scripts/activate

##### 4) Устанавливить зависимости:

    pip install -r requirements.txt

##### 5) Загрузить модель по ссылке выше.

##### 6) Запустить код:

    python main.py

Copyright (c) 2024-present, - Shnyra Bogdan
