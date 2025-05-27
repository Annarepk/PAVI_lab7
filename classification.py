from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np


def binImg(filename, resultFilename):
    with Image.open(filename) as img:
        img = img.convert('L')

        # Инвертируем цвета для поиска bbox
        invImg = ImageOps.invert(img)
        bbox = invImg.getbbox()

        if bbox:
            croppedImg = img.crop(bbox)
        else:
            croppedImg = img

        # Преобразуем в монохромный режим с порогом 128
        threshold = 128
        resImg = croppedImg.point(lambda p: 255 if p > threshold else 0, 'L')

        resImg.save(resultFilename)


def profiles(filename):
    with Image.open(filename) as img:
        width, height = img.size
        pix = img.load()

        profileX = [0] * height
        profileY = [0] * width

        for y in range(height):
            for x in range(width):
                # В монохромном изображении 0 - белый, 1 - черный
                profileX[y] += 255 - pix[x, y]
                profileY[x] += 255 - pix[x, y]

    return profileX, profileY


def segmentation(filename, hProfile, vProfile):
    with Image.open(f"{filename}/text{filename}.bmp") as img:
        width, height = img.size
        # 1. Сегментация строк (горизонтальная)
        lineSeparators = []
        thresholdH = 70  # Порог для определения пробелов между строками.
        inLine = False
        startLine = 0

        for i, val in enumerate(hProfile):
            if val > thresholdH and not inLine:
                startLine = i
                inLine = True
            elif val <= thresholdH and inLine:
                lineSeparators.append((startLine, i))
                inLine = False
        if inLine:
            lineSeparators.append((startLine, height))  # Если строка до конца

        # 2. Сегментация символов в строке (вертикальная)
        charBoxes = []
        for y1, y2 in lineSeparators:
            charSeparators = []
            thresholdV = 500  # Порог для определения пробелов между символами.
            inChar = False
            startChar = 0

            for i, val in enumerate(vProfile):
                if val > thresholdV and not inChar:
                    startChar = i
                    inChar = True
                elif val <= thresholdV and inChar:
                    charSeparators.append((startChar, i))
                    inChar = False
            if inChar:
                charSeparators.append((startChar, width))  # Если символ до конца

            # Создаем прямоугольники для символов
            for x1, x2 in charSeparators:
                charBoxes.append((x1, y1, x2, y2))

    with Image.open(f"{filename}/text{filename}.bmp") as img:
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        for x1, y1, x2, y2 in charBoxes:
            draw.rectangle((x1, y1, x2, y2), outline="red")
        img.save(f"{filename}/text{filename}Segm.png")  # Сохраняем изображение с выделенными символами
    return charBoxes


def features(img):
    pix = np.array(img)
    pix = (pix < 128).astype(np.uint8)

    if pix.sum() == 0:
        return [0] * 7  # Для пустых изображений

    # Масса
    mass = pix.sum()

    # Центр тяжести
    indices = np.argwhere(pix)
    centerY, centerX = indices.mean(axis=0)

    # Моменты инерции
    y, x = indices[:, 0], indices[:, 1]
    IXX = ((y - centerY) ** 2).sum()
    IYY = ((x - centerX) ** 2).sum()
    IXY = ((x - centerX) * (y - centerY)).sum()

    # Нормализация
    h, w = pix.shape
    size = h * w
    massNorm = mass / size
    normCentX = centerX / w
    normCentY = centerY / h
    IXXNorm = IXX / (size ** 1.5)  # Изменили степень для лучшей чувствительности
    IYYNorm = IYY / (size ** 1.5)
    IXYNorm = IXY / (size ** 1.5)

    return [massNorm, normCentX, normCentY, IXXNorm, IYYNorm, IXYNorm]


def letFromImg(binFilename, charBoxes):
    symbolFromImg = []
    with Image.open(f"{binFilename}/text{binFilename}.bmp") as img:
        img = img.convert('L')

        for i, (x1, y1, x2, y2) in enumerate(charBoxes, 1):
            symbol = img.crop((x1, y1, x2, y2))

            pix = np.array(symbol)
            coords = np.argwhere(pix < 255)

            if coords.size > 0:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1
                symbol = symbol.crop((x0, y0, x1, y1))

            symbolFromImg.append(symbol)
            symbol.save(f"letters_from_img_{binFilename}/{i}.bmp")
            binImg(f"letters_from_img_{binFilename}/{i}.bmp", f"letters_from_img_{binFilename}/{i}Bin.bmp")
    return symbolFromImg


def eucliDistance(f1, f2):
    return np.sqrt(np.sum((np.array(f1) - np.array(f2)) ** 2))


def generate_text_image(text, font_size, output_filename):
    # Выбираем шрифт (укажите путь к вашему ttf-шрифту)
    try:
        font = ImageFont.truetype("times_new_roman.ttf", font_size)
    except:
        # Если шрифт не найден, используем стандартный
        font = ImageFont.load_default()

    # Рассчитываем размер изображения
    test_img = Image.new('L', (1, 1))
    test_draw = ImageDraw.Draw(test_img)
    text_width = test_draw.textlength(text, font=font)
    text_height = font.size

    # Создаем изображение с запасом
    img = Image.new('L', (int(text_width * 1.1), int(text_height * 1.5)), color=(255))
    draw = ImageDraw.Draw(img)

    # Рисуем текст
    draw.text((10, 10), text, font=font, fill=(0))

    # Сохраняем
    img.save(output_filename)
    return output_filename


def classification(binFilename, etalonFeatures, reference):
    path = f"{binFilename}/text{binFilename}.bmp"
    hProf, vProf = profiles(path)
    charBoxes = segmentation(binFilename, hProf, vProf)

    symbolFromImg = letFromImg(binFilename, charBoxes)

    results = []
    for i, symbolImg in enumerate(symbolFromImg):
        featuresMass = features(symbolImg)

        hypotheses = []
        for letter, etalon in etalonFeatures.items():
            dist = eucliDistance(featuresMass, etalon)
            similarity = 1 / (1 + dist)
            hypotheses.append((letter, similarity))

        hypotheses.sort(key=lambda x: -x[1])
        results.append(hypotheses)

    with open(f"{binFilename}/text{binFilename}Results.txt", "w", encoding="utf-8") as f:
        for i, hypotheses in enumerate(results):
            f.write(f"{i + 1}: {hypotheses}\n")

    print("Результаты распознавания:")
    bestGuess = "".join([h[0][0] for h in results])
    print("Распознанная строка:", bestGuess)
    print("Эталонная строка:   ", reference)

    errors = sum(a != b for a, b in zip(bestGuess, reference))
    accuracy = 100 * (1 - errors / len(reference))
    print(f"Ошибок: {errors}, Точность: {accuracy:.2f}%")