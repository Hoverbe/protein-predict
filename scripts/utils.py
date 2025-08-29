def ReadFastaFile(fname: str):
    lines = open(fname).readlines()
    assert len(lines) % 2 == 0
    lines = [l.strip() for l in lines]
    headers = [lines[i * 2][1:].split()[0] for i in range(len(lines) // 2)]
    seqs = [lines[i * 2 + 1] for i in range(len(lines) // 2)]
    return headers, seqs

def SaveFastaFile(filepath, headers, seqs):
    with open(filepath, 'w') as f:
        count = 0
        for h, s in zip(headers, seqs):
            count += 1
            f.write(f'>{h}\n')
            if not count == len(headers):
                f.write(f'{s}\n')
            else:
                f.write(f'{s}')
    f.close()