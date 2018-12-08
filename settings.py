def init(number=42):
    global seed
    if (number == '') or (number == 'none'):
        seed = None
    else:
        seed = int(number)