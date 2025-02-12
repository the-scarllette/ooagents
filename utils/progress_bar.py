
def print_progress_bar(iteration: int, total: int,
                       prefix: str='', suffix: str='',
                       decimals: int=3, length: int=100,
                       fill: str='█', printed: str='\r'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printed)
    if iteration == total:
        print()
    return
