from classification import *

alphabet = "abcçdefgğhijklmnoöprsştuüvyz"
binFilename = "Bin"
reference = "sanaolanhislerimasladeğişmeyecek"


for char in alphabet:
    letterFilename = f"letters/{char}.bmp"
    with Image.open(letterFilename) as img:
        width, height = img.size
        img = img.resize((width * 2, height * 2))
        img.save(f"letters/{char}Up.bmp")
    binImg(f"letters/{char}Up.bmp", f"letters/{char}Bin.bmp")
    with Image.open(f"letters/{char}Bin.bmp") as img:
        pix = np.array(img)
        coords = np.argwhere(pix < 255)

        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            img = img.crop((x0, y0, x1, y1))
        img.save(f"letters/{char}BinCrop.bmp")

etalonFeatures = {}
for letter in alphabet:
    binLetterFilename = f"letters/{letter}BinCrop.bmp"
    with Image.open(binLetterFilename) as img:
        feats = features(img)
        etalonFeatures[letter] = feats

classification(binFilename, etalonFeatures, reference)


# Параметры эксперимента
originalFontSize = 52
newFontSize = 32

# Генерируем новое изображение

newFilename = "textNewSize.bmp"
newBinFilename = "BinNewSize"
generate_text_image(reference.replace(" ", ""), newFontSize, newFilename)

binImg(newFilename, f"{newBinFilename}/text{newBinFilename}.bmp")

print(f"\nЭксперимент с изменённым размером шрифта:")
print(f"Исходный размер: ~{originalFontSize}pt, Новый размер: {newFontSize}pt")

classification(newBinFilename, etalonFeatures, reference)

