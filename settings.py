def init(number=42):
    global seed
    if (number == '') or (number == 'none') or (number == 'None'):
        seed = None
    else:
        seed = int(number)
